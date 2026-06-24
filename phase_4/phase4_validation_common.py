"""Shared, ground-truth-free helpers for Phase 4 validation methods A-C."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import re
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPLETED_STATUSES = {"ok", "empty_output"}
OCR_MISSING = "[OCR_MISSING]"


def normalize(text: object) -> str:
    return unicodedata.normalize("NFC", " ".join(str(text or "").split()))


def natural_key(value: object) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def image_key_from_path(path: object) -> tuple[str, str, str, str] | None:
    """Return split/writer/page/image from a raw, processed, or corpus path."""
    parts = Path(str(path or "")).with_suffix("").parts
    for marker in ("processed_sample_set", "sample_set"):
        if marker in parts:
            relative = parts[parts.index(marker) + 1 :]
            if len(relative) == 4:
                return tuple(relative)
    if len(parts) >= 4:
        return tuple(parts[-4:])
    return None


def record_image_key(record: dict) -> tuple[str, str, str, str] | None:
    relative = image_key_from_path(record.get("relative_input_path"))
    if relative is not None:
        return relative
    return image_key_from_path(record.get("input_path"))


def page_key_from_image(key: tuple[str, str, str, str]) -> str:
    return "_".join(key[:3])


def image_key_sort(key: tuple[str, str, str, str]) -> tuple:
    return tuple(tuple(natural_key(part)) for part in key)


def load_latest_results(path: Path) -> tuple[dict, int, int]:
    """Load the latest JSONL record per image without consulting ground truth."""
    latest = {}
    unmapped = 0
    malformed = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            key = record_image_key(record)
            if key is None:
                unmapped += 1
                continue
            latest[key] = record
    return latest, unmapped, malformed


def load_inventory(path: Path | None) -> dict:
    """Optionally load expected image paths; transcription columns are ignored."""
    if path is None:
        return {}
    inventory = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = image_key_from_path(row.get("image_path"))
            if key is not None:
                inventory[key] = row
    return inventory


def infer_input_kind(records: dict) -> str:
    paths = [str(record.get("input_path", "")) for record in records.values()]
    if paths and all(
        any("processed" in part.lower() for part in Path(path).parts)
        for path in paths
    ):
        return "processed"
    if paths and all("sample_set" in path for path in paths):
        return "raw"
    return "mixed"


def build_pages(records: dict, inventory: dict | None = None) -> list[dict]:
    """Reconstruct ordered word-region OCR output without reading ground truth."""
    inventory = inventory or {}
    result_pages = {page_key_from_image(key) for key in records}
    pages = []
    for page_key in sorted(result_pages, key=natural_key):
        record_keys = {key for key in records if page_key_from_image(key) == page_key}
        inventory_keys = {
            key for key in inventory if page_key_from_image(key) == page_key
        }
        keys = sorted(record_keys | inventory_keys, key=image_key_sort)
        pieces = []
        completed = 0
        missing = 0
        for key in keys:
            record = records.get(key)
            if record and record.get("status") in COMPLETED_STATUSES:
                completed += 1
                text = normalize(record.get("text"))
                if text:
                    pieces.append(text)
                else:
                    pieces.append(OCR_MISSING)
                    missing += 1
            else:
                pieces.append(OCR_MISSING)
                missing += 1
        split, writer_id, page_id, _ = keys[0]
        pages.append(
            {
                "page_key": page_key,
                "split": split,
                "writer_id": writer_id,
                "page_id": page_id,
                "ocr_text": " ".join(pieces),
                # Preserve word-image boundaries. One OCR result may itself
                # contain spaces, so flattening loses information validators need.
                "ocr_regions": pieces,
                "completed_words": completed,
                "expected_words": len(keys),
                "missing_or_error_words": missing,
            }
        )
    return pages


def read_page_keys(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value.get("page_keys", []) if isinstance(value, dict) else value
        if not isinstance(values, list):
            raise ValueError("JSON page-key file must contain a page_keys list")
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip().split(",", 1)[0]
        if value and value.lower() != "page_key" and not value.startswith("#"):
            values.append(value)
    return list(dict.fromkeys(values))


def stratified_sample(pages: list[dict], count: int, seed: int) -> list[dict]:
    """Select a reproducible balanced sample across split/writer strata."""
    if count >= len(pages):
        return list(pages)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for page in pages:
        buckets.setdefault((page["split"], page["writer_id"]), []).append(page)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    strata = sorted(buckets, key=lambda value: tuple(map(str, value)))
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < count:
                selected.append(buckets[stratum].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda page: natural_key(page["page_key"]))


def select_pages(
    pages: list[dict],
    limit_pages: int | None,
    sample_pages: int | None,
    sample_seed: int,
    page_keys_path: Path | None,
) -> list[dict]:
    choices = sum(
        value is not None for value in (limit_pages, sample_pages, page_keys_path)
    )
    if choices > 1:
        raise ValueError(
            "use only one of --limit-pages, --sample-pages, or --page-keys"
        )
    if limit_pages is not None:
        if limit_pages < 1:
            raise ValueError("--limit-pages must be at least 1")
        return pages[:limit_pages]
    if sample_pages is not None:
        if sample_pages < 1:
            raise ValueError("--sample-pages must be at least 1")
        return stratified_sample(pages, sample_pages, sample_seed)
    if page_keys_path is not None:
        requested = read_page_keys(page_keys_path)
        by_key = {page["page_key"]: page for page in pages}
        missing = [key for key in requested if key not in by_key]
        if missing:
            raise ValueError(
                "requested page keys are absent from OCR results: "
                + ", ".join(missing[:10])
            )
        return [by_key[key] for key in requested]
    return pages


def response_path(output_dir: Path, page_key: str) -> Path:
    return output_dir / "responses" / f"{page_key}.json"


def load_completed_responses(
    pages: list[dict],
    output_dir: Path,
    source_results: Path,
    model: str,
    prompt_version: str,
) -> dict[str, dict]:
    completed = {}
    for page in pages:
        path = response_path(output_dir, page["page_key"])
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("status") == "ok"
            and record.get("model") == model
            and record.get("prompt_version") == prompt_version
            and record.get("source_results") == str(source_results)
            and record.get("ocr_text") == page["ocr_text"]
        ):
            completed[page["page_key"]] = record
    return completed


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def rebuild_results(path: Path, pages: list[dict], records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for page in pages:
            record = records.get(page["page_key"])
            if record is not None:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_selection(
    output_dir: Path,
    pages: list[dict],
    source_results: Path,
    sampling: dict,
) -> None:
    write_json(
        output_dir / "selected_pages.json",
        {
            "source_results": str(source_results),
            "sampling": sampling,
            "page_keys": [page["page_key"] for page in pages],
        },
    )


def usage_totals(records: list[dict]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals
