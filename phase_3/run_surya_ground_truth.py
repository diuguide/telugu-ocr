#!/usr/bin/env python3
"""Run memory-bounded Surya OCR on an image, directory, or page range."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from importlib.metadata import version
import json
import logging
from pathlib import Path
import re
import time

from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor
from surya.settings import settings

from ocr_postprocess import normalize_text, postprocess_ocr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAGES_ROOT = PROJECT_ROOT / "ground_truth/sample_set/test/9"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/evaluation/surya"
SUPPORTED_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_PARALLEL_REQUESTS = 1
DEFAULT_PROMPT_CACHE_MIB = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("surya-ground-truth")


class PlainTextParser(HTMLParser):
    """Convert Surya block HTML to plain text without external dependencies."""

    BREAK_TAGS = {"br", "div", "p", "li", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.BREAK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag.lower() in self.BREAK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        return "".join(self.parts)


def html_to_text(html):
    parser = PlainTextParser()
    parser.feed(html or "")
    parser.close()
    return normalize_text(parser.text())


def response_text(page_result):
    """Join readable Surya blocks in model-provided reading order."""
    blocks = sorted(page_result.blocks, key=lambda block: block.reading_order)
    texts = [
        html_to_text(block.html)
        for block in blocks
        if not block.skipped and not block.error and block.html
    ]
    return normalize_text(" ".join(text for text in texts if text))


def forced_text_layout(image):
    """Treat the complete word-crop image as one text-bearing OCR block."""
    width, height = image.size
    image_bbox = [0.0, 0.0, float(width), float(height)]
    text_block = LayoutBox(
        polygon=image_bbox,
        confidence=1.0,
        label="Text",
        raw_label="Text",
        position=0,
        count=50,
    )
    return LayoutResult(
        bboxes=[text_block],
        image_bbox=image_bbox,
        raw="forced single Text block for word-crop OCR",
        error=False,
    )


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    ]


def find_images(path):
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.suffix.lower() in SUPPORTED_SUFFIXES
            and not item.stem.lower().endswith((".after", ".test"))
        ),
        key=natural_key,
    )


def resolve_inputs(targets):
    """Resolve CLI targets and return images plus their relative-path root."""
    if len(targets) == 1:
        input_path = Path(targets[0]).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"input not found: {input_path}")
        images = find_images(input_path)
        relative_root = input_path.parent if input_path.is_file() else input_path
        description = input_path
    elif len(targets) in {2, 3}:
        try:
            start_page = int(targets[0])
            end_page = int(targets[1])
        except ValueError as exc:
            raise ValueError(
                "two or three positional arguments must be "
                "START_PAGE END_PAGE [PAGES_ROOT]"
            ) from exc
        if start_page < 0 or end_page < 0:
            raise ValueError("START_PAGE and END_PAGE must be non-negative")
        if start_page > end_page:
            raise ValueError("START_PAGE must be less than or equal to END_PAGE")
        pages_root = (
            Path(targets[2]).expanduser().resolve()
            if len(targets) == 3
            else DEFAULT_PAGES_ROOT
        )
        if not pages_root.is_dir():
            raise FileNotFoundError(f"pages root not found: {pages_root}")
        page_dirs = []
        for page_number in range(start_page, end_page + 1):
            page_dir = pages_root / str(page_number)
            if not page_dir.is_dir():
                raise FileNotFoundError(f"page directory not found: {page_dir}")
            page_dirs.append(page_dir)
        images = []
        for page_dir in page_dirs:
            images.extend(find_images(page_dir))
        images.sort(key=natural_key)
        relative_root = pages_root
        description = pages_root / f"{{{start_page}..{end_page}}}"
    else:
        raise ValueError(
            "provide IMAGE_FILE, DIRECTORY, or START_PAGE END_PAGE [PAGES_ROOT]"
        )

    if not images:
        raise RuntimeError(f"no supported images found: {description}")
    return images, relative_root, description


def output_paths(image_path, relative_root, output_dir):
    relative = image_path.relative_to(relative_root)
    text_path = (output_dir / "text" / relative).with_suffix(".txt")
    response_path = (output_dir / "responses" / relative).with_suffix(".json")
    return text_path, response_path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def configure_inference(args, backend):
    """Apply conservative settings before Surya starts its inference server.

    Surya's llama.cpp defaults target full-page, concurrent OCR: eight slots,
    a 98,304-token context, log probabilities, and an 8 GiB prompt cache.  This
    runner sends one small word crop at a time, so those defaults waste memory.
    In particular, the prompt cache retains every distinct image and can make a
    long run progressively slower until the OS invokes its OOM killer.
    """
    settings.SURYA_INFERENCE_PARALLEL = args.parallel_requests
    settings.SURYA_INFERENCE_LOGPROBS = args.logprobs

    if backend != "llamacpp":
        return

    settings.SURYA_INFERENCE_CTX_SIZE = args.context_size
    extra_args = (settings.LLAMA_CPP_EXTRA_ARGS or "").split()
    while "--cache-ram" in extra_args:
        index = extra_args.index("--cache-ram")
        del extra_args[index : min(index + 2, len(extra_args))]
    extra_args.extend(("--cache-ram", str(args.prompt_cache_mib)))
    settings.LLAMA_CPP_EXTRA_ARGS = " ".join(extra_args)


def completed_responses(images, relative_root, output_dir):
    """Return valid per-image responses that can be safely resumed."""
    completed = {}
    for image_path in images:
        _, response_path = output_paths(image_path, relative_root, output_dir)
        if not response_path.is_file():
            continue
        try:
            record = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable response: %s", response_path)
            continue
        if record.get("status") not in {"ok", "empty_output"}:
            continue
        if "text" not in record:
            continue
        try:
            recorded_input = Path(record["input_path"]).expanduser().resolve()
        except (KeyError, TypeError):
            continue
        if recorded_input != image_path.resolve():
            logger.warning("Ignoring response for a different input: %s", response_path)
            continue
        completed[image_path] = record
    return completed


def rebuild_results(path, records, surya_version):
    """Atomically rebuild results.jsonl from completed response records."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record in records:
            result = {**record, "surya_version": surya_version}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def process_image(
    image_path,
    relative_root,
    output_dir,
    predictor,
    transaction_number,
    transaction_count,
    min_telugu_chars,
):
    """Submit one image to Surya and persist its complete response."""
    text_path, response_path = output_paths(image_path, relative_root, output_dir)
    relative_input = image_path.relative_to(relative_root)
    logger.info(
        "[%s/%s] START %s",
        transaction_number,
        transaction_count,
        relative_input,
    )
    started = time.monotonic()
    image = None

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        layout = forced_text_layout(image)
        results = predictor([image], [layout], full_page=False)
        if len(results) != 1:
            raise RuntimeError(f"Surya returned {len(results)} results for one image")
        page_result = results[0]
        raw_text = response_text(page_result)
        processed = postprocess_ocr(raw_text, min_telugu_chars)
        text = processed.text
        elapsed = time.monotonic() - started
        status = "ok" if text else "empty_output"
        response_record = {
            "input_path": str(image_path),
            "relative_input_path": relative_input.as_posix(),
            "status": status,
            "ocr_mode": "forced_text_block",
            "elapsed_seconds": elapsed,
            "raw_text": raw_text,
            **processed.as_dict(),
            "surya_response": page_result.model_dump(mode="json"),
        }
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text + "\n", encoding="utf-8")
        write_json(response_path, response_record)
        log_method = logger.info if text else logger.warning
        log_method(
            "[%s/%s] %s %s (%.2fs, %s block(s), %s character(s))",
            transaction_number,
            transaction_count,
            "OK" if text else "EMPTY",
            relative_input,
            elapsed,
            len(page_result.blocks),
            len(text),
        )
        return response_record
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        elapsed = time.monotonic() - started
        response_record = {
            "input_path": str(image_path),
            "relative_input_path": relative_input.as_posix(),
            "status": "error",
            "elapsed_seconds": elapsed,
            "text": "",
            "suspicious_output": True,
            "suspicious_reason": "ocr_error",
            "telugu_character_count": 0,
            "error": str(exc),
        }
        write_json(response_path, response_record)
        logger.error(
            "[%s/%s] ERROR %s (%.2fs): %s",
            transaction_number,
            transaction_count,
            relative_input,
            elapsed,
            exc,
        )
        return response_record
    finally:
        if image is not None:
            image.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
  %(prog)s ground_truth/sample_set/test/9/13/1.jpg
  %(prog)s ground_truth/sample_set/test/9/13
  %(prog)s 13 15 ground_truth/sample_set/test/9

Default range root:
  {DEFAULT_PAGES_ROOT}
""",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="IMAGE_FILE, DIRECTORY, or START_PAGE END_PAGE [PAGES_ROOT]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "llamacpp"),
        help="override Surya's automatic backend selection",
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=DEFAULT_PARALLEL_REQUESTS,
        help=(
            "inference server slots (default: 1; increase only when memory allows)"
        ),
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help=(
            "llama.cpp total token context (default: 4096; sufficient for word crops)"
        ),
    )
    parser.add_argument(
        "--prompt-cache-mib",
        type=int,
        default=DEFAULT_PROMPT_CACHE_MIB,
        help=(
            "llama.cpp image-prompt cache limit in MiB (default: 0, disabled)"
        ),
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="request and store token log probabilities (slower and uses more memory)",
    )
    parser.add_argument(
        "--min-telugu-chars",
        type=int,
        default=2,
        help="flag output with fewer Telugu characters (default: 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="ignore completed response files and rerun every image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_telugu_chars < 1:
        logger.error("--min-telugu-chars must be at least 1")
        return 2
    if args.parallel_requests < 1:
        logger.error("--parallel-requests must be at least 1")
        return 2
    if args.context_size < 1024:
        logger.error("--context-size must be at least 1024")
        return 2
    if args.prompt_cache_mib < 0:
        logger.error("--prompt-cache-mib must be non-negative")
        return 2
    try:
        images, relative_root, description = resolve_inputs(args.targets)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "run_summary.json"

    surya_version = version("surya-ocr")
    logger.info("Input: %s", description)
    logger.info("Images: %s", len(images))
    logger.info("Output: %s", output_dir)
    logger.info("Surya OCR: %s", surya_version)

    # Construction is lazy: this selects a backend without loading the model.
    # Configure its server limits before the first prediction starts it.
    manager = SuryaInferenceManager(method=args.backend)
    backend = manager.method
    configure_inference(args, backend)
    logger.info(
        "Inference: backend=%s, parallel=%s, context=%s, prompt-cache=%s MiB, "
        "logprobs=%s",
        backend,
        args.parallel_requests,
        args.context_size if backend == "llamacpp" else "backend default",
        args.prompt_cache_mib if backend == "llamacpp" else "backend default",
        args.logprobs,
    )

    completed = (
        {} if args.overwrite else completed_responses(images, relative_root, output_dir)
    )
    pending = [image for image in images if image not in completed]
    rebuild_results(
        results_path,
        (completed[image] for image in images if image in completed),
        surya_version,
    )
    logger.info("Resume: %s complete, %s pending", len(completed), len(pending))

    successful = sum(record["status"] == "ok" for record in completed.values())
    failed = sum(
        record["status"] != "ok" for record in completed.values()
    )
    run_started = time.monotonic()
    try:
        if pending:
            predictor = RecognitionPredictor(manager)
            image_numbers = {
                image: number for number, image in enumerate(images, start=1)
            }
            with results_path.open("a", encoding="utf-8") as results_file:
                for image_path in pending:
                    result = process_image(
                        image_path,
                        relative_root,
                        output_dir,
                        predictor,
                        image_numbers[image_path],
                        len(images),
                        args.min_telugu_chars,
                    )
                    result["surya_version"] = surya_version
                    results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results_file.flush()
                    if result["status"] == "ok":
                        successful += 1
                    else:
                        failed += 1
    except KeyboardInterrupt:
        logger.warning("Interrupted; completed responses have been preserved.")
        return 130
    finally:
        if manager is not None:
            manager.stop()

    elapsed = time.monotonic() - run_started
    summary = {
        "input": str(description),
        "image_count": len(images),
        "resumed": len(completed),
        "processed_this_run": len(pending),
        "successful": successful,
        "failed": failed,
        "elapsed_seconds": elapsed,
        "surya_version": surya_version,
        "backend": backend,
        "parallel_requests": args.parallel_requests,
        "context_size": args.context_size if backend == "llamacpp" else None,
        "prompt_cache_mib": (
            args.prompt_cache_mib if backend == "llamacpp" else None
        ),
        "logprobs": args.logprobs,
    }
    write_json(summary_path, summary)
    logger.info(
        "DONE: %s successful, %s failed, %.2fs total",
        successful,
        failed,
        elapsed,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
