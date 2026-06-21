#!/usr/bin/env python3
"""Binarize PNG images with Otsu thresholding."""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_png(path: Path) -> np.ndarray:
    """Load a PNG as grayscale, flattening transparency onto white."""
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            foreground = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, foreground)

        return np.asarray(image.convert("L"))


def binarize(image: np.ndarray) -> np.ndarray:
    """Binarize a grayscale or RGB image using Otsu's global threshold."""
    gray = (
        image
        if image.ndim == 2
        else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    )
    _, thresholded = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresholded


def save_png(image: np.ndarray, path: Path) -> None:
    """Save a binary grayscale numpy image as PNG."""
    Image.fromarray(image).save(path, "PNG")


def test_output_path(path: Path) -> Path:
    """Return the side-by-side test output path for a PNG."""
    return path.with_name(f"{path.stem}.test{path.suffix}")


def process_png(path: Path, test: bool) -> bool:
    """Binarize one PNG, overwriting it unless test mode is enabled."""
    output_path = test_output_path(path) if test else path

    try:
        image = load_png(path)
        thresholded = binarize(image)
        save_png(thresholded, output_path)
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)
        return False

    logger.info("Processed %s -> %s", path, output_path)
    return True


def find_pngs(path: Path) -> list[Path]:
    """Find PNG files from a single image path or a directory tree."""
    if path.is_file():
        return [path] if path.suffix.lower() == ".png" else []

    return sorted(
        item
        for item in path.rglob("*.png")
        if item.is_file() and not item.name.endswith(".test.png")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binarize PNG images with Otsu thresholding."
    )
    parser.add_argument("input", help="PNG image path or directory to scan")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Write name.test.png beside each source image instead of overwriting it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        return 1

    pngs = find_pngs(input_path)
    if not pngs:
        logger.error("No PNG files found: %s", input_path)
        return 1

    successful = sum(1 for png in pngs if process_png(png, args.test))
    failed = len(pngs) - successful

    logger.info("Done: %s successful, %s failed", successful, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
