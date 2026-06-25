#!/usr/bin/env python3
"""Build OCR ground-truth labels and a manifest from the selected TSV file."""

from collections import Counter, defaultdict
import csv
from pathlib import Path
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SELECTED_FILE = PROJECT_ROOT / "bin/phase_1/selected_ground_truth_pages.txt"
WORD_LABEL_DIR = PROJECT_ROOT / "data/ground_truth/word_labels"
PAGE_LABEL_DIR = PROJECT_ROOT / "data/ground_truth/page_labels"
MANIFEST_DIR = PROJECT_ROOT / "bin/phase_1/ground_truth/manifests"
MANIFEST_FILE = MANIFEST_DIR / "ground_truth_manifest.csv"
LOG_DIR = PROJECT_ROOT / "logs"
WARNING_FILE = LOG_DIR / "ground_truth_warnings.csv"

MANIFEST_FIELDS = [
    "page_key",
    "split",
    "writer_id",
    "page_id",
    "image_index",
    "image_path",
    "vocab_id",
    "ground_truth_text",
    "word_label_path",
    "page_label_path",
]
WARNING_FIELDS = ["issue", "image_path", "vocab_id", "label"]


def normalized_label(value):
    return unicodedata.normalize("NFC", (value or "").strip())


def parse_image_path(value):
    """Return split, writer, page, and numeric image index, or None."""
    path = Path(value)
    if len(path.parts) < 4 or path.suffix.lower() != ".jpg":
        return None

    split, writer_id, page_id = path.parts[-4:-1]
    image_stem = path.stem
    if not (
        split
        and writer_id.isdigit()
        and page_id.isdigit()
        and image_stem.isdigit()
    ):
        return None

    return split, writer_id, page_id, int(image_stem)


def relative_text(path):
    return path.relative_to(PROJECT_ROOT).as_posix()


def main():
    for directory in (WORD_LABEL_DIR, PAGE_LABEL_DIR, MANIFEST_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    loaded_rows = 0
    word_files_written = 0
    warnings = []
    issue_counts = Counter()
    records = []
    pages = defaultdict(list)

    with SELECTED_FILE.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required_fields = {"image_path", "vocab_id", "label"}
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                "selected file must contain tab-separated image_path, vocab_id, "
                "and label columns"
            )

        for row in reader:
            loaded_rows += 1
            image_path = (row.get("image_path") or "").strip()
            vocab_id = (row.get("vocab_id") or "").strip()
            label = normalized_label(row.get("label"))
            parsed = parse_image_path(image_path)

            row_issues = []
            if parsed is None:
                row_issues.append("malformed_path")
            if image_path and not (PROJECT_ROOT / image_path).is_file():
                row_issues.append("missing_image")
            if not label:
                row_issues.append("empty_label")
            if not vocab_id:
                row_issues.append("missing_vocab_id")

            for issue in row_issues:
                issue_counts[issue] += 1
                warnings.append(
                    {
                        "issue": issue,
                        "image_path": image_path,
                        "vocab_id": vocab_id,
                        "label": label,
                    }
                )

            # A missing image is retained in the manifest so the checker can
            # report it. Other invalid fields prevent a clean label record.
            if parsed is None or not label or not vocab_id:
                continue

            split, writer_id, page_id, image_index = parsed
            page_key = f"{split}_{writer_id}_{page_id}"
            word_path = WORD_LABEL_DIR / f"{page_key}_{image_index}.txt"
            page_path = PAGE_LABEL_DIR / f"{page_key}.txt"
            word_path.write_text(label + "\n", encoding="utf-8")
            word_files_written += 1

            record = {
                "page_key": page_key,
                "split": split,
                "writer_id": writer_id,
                "page_id": page_id,
                "image_index": str(image_index),
                "image_path": image_path,
                "vocab_id": vocab_id,
                "ground_truth_text": label,
                "word_label_path": relative_text(word_path),
                "page_label_path": relative_text(page_path),
            }
            records.append(record)
            pages[page_key].append(record)

    def record_sort_key(record):
        split_order = {"train": 0, "val": 1, "test": 2}
        return (
            split_order.get(record["split"], 3),
            record["split"],
            int(record["writer_id"]),
            int(record["page_id"]),
            int(record["image_index"]),
        )

    records.sort(key=record_sort_key)

    for page_records in pages.values():
        page_records.sort(key=lambda record: int(record["image_index"]))
        page_path = PROJECT_ROOT / page_records[0]["page_label_path"]
        page_text = "".join(
            record["ground_truth_text"] + "\n" for record in page_records
        )
        page_path.write_text(page_text, encoding="utf-8")

    with MANIFEST_FILE.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=MANIFEST_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    with WARNING_FILE.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=WARNING_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(warnings)

    print(f"Loaded rows: {loaded_rows}")
    print(f"Word label files written: {word_files_written}")
    print(f"Page label files written: {len(pages)}")
    print(f"Manifest written to: {relative_text(MANIFEST_FILE)}")
    print(f"Warnings written to: {relative_text(WARNING_FILE)}")
    print(f"Missing images: {issue_counts['missing_image']}")
    print(f"Empty labels: {issue_counts['empty_label']}")
    print(f"Missing vocab IDs: {issue_counts['missing_vocab_id']}")
    print(f"Malformed paths: {issue_counts['malformed_path']}")


if __name__ == "__main__":
    main()
