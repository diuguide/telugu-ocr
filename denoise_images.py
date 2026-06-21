#!/usr/bin/env python3
"""Remove noise and small speckles from standardized PNG images."""

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
    """Load a PNG as a grayscale numpy array."""
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            mask = image.split()[-1] if image.mode == "RGBA" else None
            background.paste(image, mask=mask)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        rgb = np.asarray(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def remove_small_components(
    image: np.ndarray,
    foreground_threshold: int,
    min_component_area: int,
) -> np.ndarray:
    """Remove tiny dark foreground components from a mostly white binary image."""
    if min_component_area <= 0:
        return image

    foreground = image < foreground_threshold
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=8,
    )

    cleaned = image.copy()
    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_component_area:
            cleaned[labels == label] = 255

    return cleaned


def normalize_background(image: np.ndarray, sigma: float) -> np.ndarray:
    """Flatten slowly varying paper texture while preserving dark foreground."""
    if sigma <= 0:
        return image

    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return cv2.divide(image, background, scale=255)


def denoise_and_despeckle(
    image: np.ndarray,
    h: float,
    min_component_area: int,
    foreground_threshold: int,
    background_sigma: float = 75,
) -> np.ndarray:
    """Denoise, flatten the paper background, then remove dark speckles."""
    denoised = cv2.fastNlMeansDenoising(image, h=h)
    normalized = normalize_background(denoised, sigma=background_sigma)
    return remove_small_components(
        normalized,
        foreground_threshold=foreground_threshold,
        min_component_area=min_component_area,
    )


def save_png(image: np.ndarray, path: Path) -> None:
    """Save a grayscale numpy image as PNG."""
    Image.fromarray(image).save(path, "PNG")


def test_output_path(path: Path) -> Path:
    """Return the side-by-side test output path for an image."""
    return path.with_name(f"{path.stem}.test{path.suffix}")


def process_png(
    path: Path,
    test: bool,
    h: float,
    min_component_area: int,
    foreground_threshold: int,
    background_sigma: float,
) -> bool:
    """Denoise one PNG, overwriting it unless test mode is enabled."""
    output_path = test_output_path(path) if test else path

    try:
        image = load_png(path)
        cleaned = denoise_and_despeckle(
            image,
            h=h,
            min_component_area=min_component_area,
            foreground_threshold=foreground_threshold,
            background_sigma=background_sigma,
        )
        save_png(cleaned, output_path)
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)
        return False

    logger.info("Processed %s -> %s", path, output_path)
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
        description="Remove noise and small speckles from standardized PNG images."
    )
    parser.add_argument("input", help="PNG image path or directory to scan")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Write name.test.png beside each source image instead of overwriting it",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=10,
        help="NL-means denoising strength passed to cv2.fastNlMeansDenoising",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=70,
        help="Remove dark components smaller than this pixel area (default: 70)",
    )
    parser.add_argument(
        "--background-sigma",
        type=float,
        default=75,
        help="Scale of paper-background normalization; 0 disables it (default: 75)",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=128,
        help="Pixels below this value are treated as foreground speckles",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if args.h < 0:
        logger.error("Denoising strength must be non-negative: %s", args.h)
        return 1

    if args.min_component_area < 0:
        logger.error(
            "Minimum component area must be non-negative: %s",
            args.min_component_area,
        )
        return 1

    if args.background_sigma < 0:
        logger.error(
            "Background sigma must be non-negative: %s",
            args.background_sigma,
        )
        return 1

    if not 0 <= args.foreground_threshold <= 255:
        logger.error(
            "Foreground threshold must be between 0 and 255: %s",
            args.foreground_threshold,
        )
        return 1

    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        return 1

    pngs = find_pngs(input_path)
    if not pngs:
        logger.error("No PNG files found: %s", input_path)
        return 1

    successful = sum(
        1
        for png in pngs
        if process_png(
            png,
            test=args.test,
            h=args.h,
            min_component_area=args.min_component_area,
            foreground_threshold=args.foreground_threshold,
            background_sigma=args.background_sigma,
        )
    )
    failed = len(pngs) - successful

    logger.info("Done: %s successful, %s failed", successful, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
