#!/usr/bin/env python3
"""Enhance standardized PNG contrast for OCR preprocessing."""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, ImageEnhance


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_png(path: Path) -> Image.Image:
    """Load a PNG as an RGB PIL image."""
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            mask = image.split()[-1] if image.mode == "RGBA" else None
            background.paste(image, mask=mask)
            return background

        if image.mode != "RGB":
            return image.convert("RGB")

        return image.copy()


def enhance_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Enhance contrast globally; the factor is independent of image dimensions."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def save_png(image: Image.Image, path: Path) -> None:
    """Save a PIL image as PNG."""
    image.save(path, "PNG")


def test_output_path(path: Path) -> Path:
    """Return the side-by-side test output path for an image."""
    return path.with_name(f"{path.stem}.test{path.suffix}")


def process_png(path: Path, test: bool, factor: float) -> bool:
    """Enhance one PNG, overwriting it unless test mode is enabled."""
    output_path = test_output_path(path) if test else path

    try:
        image = load_png(path)
        enhanced = enhance_contrast(image, factor)
        save_png(enhanced, output_path)
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)
        return False

    logger.info("Processed %s -> %s (factor: %.2f)", path, output_path, factor)
    return True


def find_pngs(path: Path) -> list[Path]:
    """Find PNG files from a single image path or directory tree."""
    if path.is_file():
        return [path] if path.suffix.lower() == ".png" else []

    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() == ".png"
        and not item.stem.lower().endswith(".test")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance standardized PNG contrast for OCR preprocessing."
    )
    parser.add_argument("input", help="PNG image path or directory to scan")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Write name.test.png beside each source image instead of overwriting it",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=2.0,
        help="Global, dimension-independent contrast multiplier (default: 2.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if args.factor < 0:
        logger.error("Contrast factor must be non-negative: %s", args.factor)
        return 1

    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        return 1

    pngs = find_pngs(input_path)
    if not pngs:
        logger.error("No PNG files found: %s", input_path)
        return 1

    successful = sum(
        1 for png in pngs if process_png(png, args.test, args.factor)
    )
    failed = len(pngs) - successful

    logger.info("Done: %s successful, %s failed", successful, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
