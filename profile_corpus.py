#!/usr/bin/env python3
"""Profile the Telugu OCR corpus inventory and image quality."""

import argparse
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from count_docs_pages import (
    SPLITS,
    count_documents_pages_and_writers,
    count_manifest_items_and_characters,
    count_root_file_types,
    merge_file_type_counts,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def summarize(values: list[float]) -> dict[str, float | None]:
    """Return basic descriptive statistics for a numeric series."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def format_number(value: float | None) -> str:
    """Format optional numeric statistics for display."""
    if value is None:
        return "N/A"

    return f"{value:.2f}"


def print_stat_block(name: str, stats_dict: dict[str, float | None]) -> None:
    """Print a small statistic block."""
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Count: {stats_dict['count']}")

    if stats_dict["count"] == 0:
        print("No data available")
        return

    print(f"Mean : {format_number(stats_dict['mean'])}")
    print(f"Std  : {format_number(stats_dict['std'])}")
    print(f"Min  : {format_number(stats_dict['min'])}")
    print(f"Max  : {format_number(stats_dict['max'])}")


def telugu_seg_root(dataset_root: Path) -> Path:
    """Return the TeluguSeg directory for a dataset root."""
    root = dataset_root / "TeluguSeg"
    if not root.exists():
        raise FileNotFoundError(f"Could not find {root}")

    return root


def iter_segmented_images(dataset_root: Path, max_images: int | None = None) -> list[Path]:
    """Find segmented OCR image files under TeluguSeg split folders."""
    root = telugu_seg_root(dataset_root)
    image_paths: list[Path] = []

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            logger.info("Skipping missing split: %s", split)
            continue

        for image_path in split_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(image_path)
                if max_images is not None and len(image_paths) >= max_images:
                    return sorted(image_paths)

    return sorted(image_paths)


def profile_images(image_paths: list[Path]) -> dict[str, object]:
    """Compute resolution and scan-quality metrics for corpus images."""
    resolution_counter: Counter[tuple[int, int]] = Counter()
    brightness_values: list[float] = []
    contrast_values: list[float] = []
    blur_values: list[float] = []
    unreadable_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            unreadable_count += 1
            logger.warning("Could not read image: %s", image_path)
            continue

        height, width = image.shape
        resolution_counter[(width, height)] += 1
        brightness_values.append(float(np.mean(image)))
        contrast_values.append(float(np.std(image)))
        blur_values.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))

    return {
        "image_count": len(image_paths),
        "readable_image_count": len(image_paths) - unreadable_count,
        "unreadable_image_count": unreadable_count,
        "resolution_counter": resolution_counter,
        "brightness_stats": summarize(brightness_values),
        "contrast_stats": summarize(contrast_values),
        "blur_stats": summarize(blur_values),
    }


def print_inventory(dataset_root: Path) -> None:
    """Print document/page/character counts from the corpus inventory."""
    root_file_type_counts = count_root_file_types(dataset_root)
    segmented_file_type_counts, page_count, writer_count = (
        count_documents_pages_and_writers(dataset_root)
    )
    manifest_item_count, manifest_character_count = (
        count_manifest_items_and_characters(dataset_root)
    )
    total_file_type_counts = merge_file_type_counts(
        segmented_file_type_counts,
        root_file_type_counts,
    )
    segmented_document_count = sum(segmented_file_type_counts.values())
    root_level_document_count = sum(root_file_type_counts.values())
    document_count = sum(total_file_type_counts.values())

    print("Corpus inventory")
    print("----------------")
    print(f"Document count: {document_count}")
    print(f"  Segmented page-level files: {segmented_document_count}")
    print(f"  Root-level files: {root_level_document_count}")
    print(f"Writer count: {writer_count}")
    print(f"Page count: {page_count}")
    print(f"Manifest labeled items: {manifest_item_count}")
    print(f"Estimated total characters: {manifest_character_count:,}")

    print("\nCombined file type counts")
    print("-------------------------")
    if not total_file_type_counts:
        print("No files found")
        return

    for file_type, count in sorted(total_file_type_counts.items()):
        print(f"{file_type}: {count}")


def print_image_profile(image_profile: dict[str, object], top_resolutions: int) -> None:
    """Print image resolution and scan-quality profile."""
    resolution_counter = image_profile["resolution_counter"]

    if not isinstance(resolution_counter, Counter):
        raise TypeError("resolution_counter must be a Counter")

    print("\nImage profile")
    print("-------------")
    print(f"Image files scanned: {image_profile['image_count']}")
    print(f"Readable images: {image_profile['readable_image_count']}")
    print(f"Unreadable images: {image_profile['unreadable_image_count']}")

    print(f"\nImage resolution distribution (top {top_resolutions})")
    print("-----------------------------------")
    if not resolution_counter:
        print("No image resolutions available")
    else:
        for (width, height), count in resolution_counter.most_common(top_resolutions):
            print(f"{width}x{height}: {count}")

    brightness_stats = image_profile["brightness_stats"]
    contrast_stats = image_profile["contrast_stats"]
    blur_stats = image_profile["blur_stats"]

    if not isinstance(brightness_stats, dict):
        raise TypeError("brightness_stats must be a dict")
    if not isinstance(contrast_stats, dict):
        raise TypeError("contrast_stats must be a dict")
    if not isinstance(blur_stats, dict):
        raise TypeError("blur_stats must be a dict")

    print_stat_block("Brightness statistics", brightness_stats)
    print_stat_block("Contrast statistics", contrast_stats)
    print_stat_block("Blur statistics (variance of Laplacian)", blur_stats)

    print("\nScan quality variability")
    print("------------------------")
    if blur_stats["count"] == 0:
        print("Unavailable")
        return

    print("Estimated from per-image brightness, contrast, and blur dispersion.")
    print(f"Brightness std: {format_number(brightness_stats['std'])}")
    print(f"Contrast std  : {format_number(contrast_stats['std'])}")
    print(f"Blur std      : {format_number(blur_stats['std'])}")


def print_historical_period(historical_period: str | None) -> None:
    """Print known historical period metadata, if provided."""
    print("\nHistorical period")
    print("-----------------")
    print(historical_period or "Unknown from available corpus metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the Telugu OCR corpus inventory and image quality."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/tom/telugu_data"),
        help="Dataset root path (default: /home/tom/telugu_data)",
    )
    parser.add_argument(
        "--historical-period",
        help="Known historical period to include in the profile report",
    )
    parser.add_argument(
        "--top-resolutions",
        type=int,
        default=10,
        help="Number of resolution buckets to print",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Only profile this many images for a faster exploratory scan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.root.exists():
        logger.error("Dataset root not found: %s", args.root)
        return 1

    if args.top_resolutions < 1:
        logger.error("--top-resolutions must be at least 1")
        return 1

    if args.max_images is not None and args.max_images < 1:
        logger.error("--max-images must be at least 1")
        return 1

    logger.info("Profiling corpus under %s", args.root)

    try:
        image_paths = iter_segmented_images(args.root, max_images=args.max_images)
        image_profile = profile_images(image_paths)

        print("\n========== CORPUS PROFILE ==========\n")
        print_inventory(args.root)
        print_image_profile(image_profile, top_resolutions=args.top_resolutions)
        print_historical_period(args.historical_period)
    except Exception as exc:
        logger.error("Failed to profile corpus: %s", exc)
        return 1

    logger.info("Finished corpus profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
