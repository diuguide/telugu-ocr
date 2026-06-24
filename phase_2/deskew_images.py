#!/usr/bin/env python3
"""Detect skew in PNG images and rotate them in place."""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from deskew import determine_skew
from skimage.transform import rotate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MIN_SKEW_ANGLE = -15.0
MAX_SKEW_ANGLE = 15.0


def load_png(path: Path) -> np.ndarray:
    """Load a PNG, preserving grayscale images and flattening transparency."""
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            foreground = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, foreground).convert("RGB")
        elif image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        return np.asarray(image)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a skimage float image back to uint8 for saving."""
    if image.dtype == np.uint8:
        return image

    return (np.clip(image, 0, 1) * 255).round().astype(np.uint8)


def threshold_for_skew_detection(image: np.ndarray) -> np.ndarray:
    """Create a high-contrast binary image for skew detection."""
    grayscale = (
        image
        if image.ndim == 2
        else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    )
    _, thresholded = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return thresholded


def rotate_by_detected_skew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Determine the image skew angle and rotate by that angle."""
    thresholded = threshold_for_skew_detection(image)
    angle = determine_skew(
        thresholded,
        min_angle=MIN_SKEW_ANGLE,
        max_angle=MAX_SKEW_ANGLE,
    )
    angle = float(angle) if angle else 0.0

    if angle == 0.0:
        return image, angle

    rotated = rotate(
        image,
        angle,
        resize=True,
        cval=1,
        mode="constant",
    )
    return to_uint8(rotated), angle


def save_png(image: np.ndarray, path: Path) -> None:
    """Save an RGB numpy image as PNG."""
    Image.fromarray(to_uint8(image)).save(path, "PNG")


def test_output_path(path: Path) -> Path:
    """Return the side-by-side test output path for a PNG."""
    return path.with_name(f"{path.stem}.test{path.suffix}")


def process_png(path: Path, test: bool) -> bool:
    """Deskew one PNG, overwriting it unless test mode is enabled."""
    output_path = test_output_path(path) if test else path

    try:
        image = load_png(path)
        rotated, angle = rotate_by_detected_skew(image)
        save_png(rotated, output_path)
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)
        return False

    logger.info("Processed %s -> %s (angle: %.2f)", path, output_path, angle)
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
        description="Detect skew in PNG images and rotate them in place."
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
