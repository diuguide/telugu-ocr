#!/usr/bin/env python3
"""Run Tesseract Telugu OCR on an image, a directory, or page range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from ocr_postprocess import normalize_text, postprocess_ocr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAGES_ROOT = PROJECT_ROOT / "ground_truth/sample_set/"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/evaluation/tesseract"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.as_posix())
    ]


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        (
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=natural_key,
    )


def resolve_inputs(targets: list[str]) -> tuple[list[Path], Path, Path]:
    """Resolve an image, directory, or inclusive page range."""
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
        images = []
        for page_number in range(start_page, end_page + 1):
            page_dir = pages_root / str(page_number)
            if not page_dir.is_dir():
                raise FileNotFoundError(f"page directory not found: {page_dir}")
            images.extend(find_images(page_dir))
        images.sort(key=natural_key)
        relative_root = pages_root
        description = pages_root / f"{{{start_page}..{end_page}}}"
    else:
        raise ValueError(
            "provide IMAGE_FILE, DIRECTORY, or START_PAGE END_PAGE [PAGES_ROOT]"
        )

    if not images:
        raise RuntimeError(f"no JPG or PNG images found: {description}")
    return images, relative_root, description


def check_tesseract(command: str, language: str) -> tuple[str, str]:
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(f"Tesseract executable not found: {command}")

    version = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout.splitlines()[0]
    language_output = subprocess.run(
        [executable, "--list-langs"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout
    installed_languages = set(language_output.splitlines())
    if language not in installed_languages:
        raise RuntimeError(f"Tesseract language pack is not installed: {language}")
    return executable, version


def output_paths(image: Path, input_root: Path, output_dir: Path) -> tuple[Path, Path]:
    relative = image.relative_to(input_root)
    return (
        (output_dir / "text" / relative).with_suffix(".txt"),
        (output_dir / "responses" / relative).with_suffix(".json"),
    )


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def completed_responses(
    images: list[Path],
    input_root: Path,
    output_dir: Path,
    language: str,
    psm: int,
    oem: int,
) -> dict[Path, dict[str, object]]:
    """Return valid responses produced with the requested OCR settings."""
    completed = {}
    for image in images:
        text_path, response_path = output_paths(image, input_root, output_dir)
        if not response_path.is_file() or not text_path.is_file():
            continue
        try:
            record = json.loads(response_path.read_text(encoding="utf-8"))
            recorded_input = Path(str(record["input_path"])).expanduser().resolve()
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if record.get("status") not in {"ok", "empty_output"}:
            continue
        if "text" not in record or recorded_input != image.resolve():
            continue
        if (
            record.get("language") != language
            or record.get("psm") != psm
            or record.get("oem") != oem
        ):
            continue
        completed[image] = record
    return completed


def rebuild_results(
    path: Path,
    images: list[Path],
    completed: dict[Path, dict[str, object]],
) -> None:
    """Atomically rebuild results.jsonl from resumable response files."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for image in images:
            if image in completed:
                output.write(json.dumps(completed[image], ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def recognize_image(
    image: Path,
    executable: str,
    language: str,
    psm: int,
    oem: int,
    timeout: float,
    min_telugu_chars: int,
) -> dict[str, object]:
    """Send one image to Tesseract and return its Telugu text response."""
    command = [
        executable,
        str(image),
        "stdout",
        "-l",
        language,
        "--psm",
        str(psm),
        "--oem",
        str(oem),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            normalize_text(result.stderr)
            or f"Tesseract exited with status {result.returncode}"
        )

    processed = postprocess_ocr(result.stdout, min_telugu_chars)
    return {
        "status": "empty_output" if not processed.text else "ok",
        "raw_text": normalize_text(result.stdout),
        **processed.as_dict(),
        "elapsed_seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language", default="Telugu")
    parser.add_argument("--psm", type=int, default=8, help="page segmentation mode")
    parser.add_argument("--oem", type=int, default=1, help="OCR engine mode")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-telugu-chars", type=int, default=2)
    parser.add_argument("--tesseract-command", default="tesseract")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="ignore completed response files and rerun every image",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.min_telugu_chars < 1:
        raise ValueError("--min-telugu-chars must be at least 1")

    images, input_root, description = resolve_inputs(args.targets)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executable, tesseract_version = check_tesseract(
        args.tesseract_command, args.language
    )
    print(f"Input: {description}")
    print(f"Images: {len(images)}")
    print(f"Output: {output_dir}")

    results_path = output_dir / "results.jsonl"
    completed = (
        {}
        if args.overwrite
        else completed_responses(
            images,
            input_root,
            output_dir,
            args.language,
            args.psm,
            args.oem,
        )
    )
    pending = [image for image in images if image not in completed]
    rebuild_results(results_path, images, completed)
    print(f"Resume: {len(completed)} complete, {len(pending)} pending")

    failures = 0
    image_numbers = {image: number for number, image in enumerate(images, start=1)}
    with results_path.open("a", encoding="utf-8") as results_file:
        for image in pending:
            number = image_numbers[image]
            relative_input = image.relative_to(input_root)
            try:
                response = recognize_image(
                    image,
                    executable,
                    args.language,
                    args.psm,
                    args.oem,
                    args.timeout,
                    args.min_telugu_chars,
                )
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                failures += 1
                response = {
                    "status": "error",
                    "text": "",
                    "suspicious_output": True,
                    "suspicious_reason": "ocr_error",
                    "telugu_character_count": 0,
                    "error": str(exc),
                }

            record = {
                "input_path": str(image),
                "relative_input_path": relative_input.as_posix(),
                **response,
                "tesseract_version": tesseract_version,
                "language": args.language,
                "psm": args.psm,
                "oem": args.oem,
            }
            text_path, response_path = output_paths(image, input_root, output_dir)
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(str(record["text"]) + "\n", encoding="utf-8")
            write_json(response_path, record)
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_file.flush()
            print(
                f"[{number}/{len(images)}] {relative_input}: "
                f"{record['status']} — {record['text']}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
