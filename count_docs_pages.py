from pathlib import Path
import argparse
import logging


SPLITS = ("train", "val", "test")


LOGGER = logging.getLogger(__name__)


def character_count_from_split_file(split_file: Path) -> tuple[int, int]:
    """
    Each line format:
    relative/image/path.jpg transcription

    Returns:
    - labeled item count
    - total character count from transcriptions
    """
    labeled_items = 0
    total_characters = 0

    with open(split_file, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            _, label = parts
            labeled_items += 1
            total_characters += len(label.strip())

    return labeled_items, total_characters


def count_manifest_items_and_characters(dataset_root: Path) -> tuple[int, int]:
    """
    Sum labeled items and characters from split manifest files at dataset root.
    """
    manifest_item_count = 0
    manifest_character_count = 0

    LOGGER.info("Scanning split manifests under %s", dataset_root)

    for split in SPLITS:
        split_manifest = dataset_root / f"{split}.txt"
        if not split_manifest.exists():
            LOGGER.info("Skipping missing manifest: %s", split_manifest.name)
            continue

        LOGGER.info("Processing manifest: %s", split_manifest.name)
        labeled_items, split_characters = character_count_from_split_file(split_manifest)
        manifest_item_count += labeled_items
        manifest_character_count += split_characters

    return manifest_item_count, manifest_character_count


def count_root_file_types(dataset_root: Path) -> dict[str, int]:
    """
    Count file extensions for files directly under the dataset root directory.
    """
    file_type_counts: dict[str, int] = {}

    for entry in dataset_root.iterdir():
        if not entry.is_file():
            continue

        suffix = entry.suffix.lower()
        key = suffix if suffix else "[no_extension]"
        file_type_counts[key] = file_type_counts.get(key, 0) + 1

    return file_type_counts


def count_documents_and_pages(dataset_root: Path) -> tuple[int, int]:
    """
    Count pages and documents under TeluguSeg split folders.

    Expected structure:
    TeluguSeg/<split>/<writer_id>/<page_id>/

    Page count rule:
    - For each writer, find numeric page directory names and take the maximum.
    - Sum those maxima across all writers.

    Document count rule:
    - For each writer, inspect page directories from 1..max_page_id.
    - Count bottom-level .jpg and .txt files in those page directories.
    """
    telugu_seg_root = dataset_root / "TeluguSeg"
    if not telugu_seg_root.exists():
        raise FileNotFoundError(f"Could not find {telugu_seg_root}")

    document_count = 0
    page_count = 0

    LOGGER.info("Scanning split directories under %s", telugu_seg_root)

    for split in SPLITS:
        split_dir = telugu_seg_root / split
        if not split_dir.exists():
            LOGGER.info("Skipping missing split: %s", split)
            continue

        LOGGER.info("Processing split: %s", split)

        for writer_dir in split_dir.iterdir():
            if not writer_dir.is_dir():
                continue

            page_ids = []
            for page_dir in writer_dir.iterdir():
                if not page_dir.is_dir():
                    continue
                if page_dir.name.isdigit():
                    page_ids.append(int(page_dir.name))

            if not page_ids:
                continue

            max_page_id = max(page_ids)
            page_count += max_page_id

            for page_id in range(1, max_page_id + 1):
                page_dir = writer_dir / str(page_id)
                if not page_dir.is_dir():
                    continue

                for file_path in page_dir.iterdir():
                    if not file_path.is_file():
                        continue
                    if file_path.suffix.lower() in {".jpg", ".txt"}:
                        document_count += 1

    return document_count, page_count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Count documents and pages in the Telugu dataset"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/tom/telugu_data"),
        help="Dataset root path (default: /home/tom/telugu_data)",
    )
    args = parser.parse_args()

    LOGGER.info("Starting dataset count job for root: %s", args.root)

    file_type_counts = count_root_file_types(args.root)
    document_count, page_count = count_documents_and_pages(args.root)
    manifest_item_count, manifest_character_count = count_manifest_items_and_characters(args.root)

    LOGGER.info("Finished dataset count job")

    print(f"Document count (.jpg + .txt files): {document_count}")
    print(f"Page count (sum of max page-dir per writer): {page_count}")
    print(f"Manifest labeled items (train/val/test .txt): {manifest_item_count}")
    print(f"Estimated total characters from manifests: {manifest_character_count:,}")

    print("\nRoot-level file type counts:")
    if not file_type_counts:
        print("No files found directly under dataset root")
    else:
        for file_type, count in sorted(file_type_counts.items()):
            print(f"{file_type}: {count}")


if __name__ == "__main__":
    main()
