#!/usr/bin/env python3
"""Conservatively crop images for Telugu OCR preprocessing."""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FINAL_BORDER_CLIP = 15


@dataclass(frozen=True)
class CropBox:
    """Pixel crop bounds with an exclusive right and bottom edge."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class CropSettings:
    """Cropping controls tuned for OCR-safe margin removal."""

    dark_delta: int
    min_component_area: int
    padding: int
    dilate: int
    remove_edge_components: bool
    min_crop_ratio: float


def load_png(path: Path) -> np.ndarray:
    """Load a PNG as an RGB numpy array."""
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            mask = image.split()[-1] if image.mode == "RGBA" else None
            background.paste(image, mask=mask)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return np.asarray(image)


def save_png(image: np.ndarray, path: Path) -> None:
    """Save an RGB numpy image as PNG."""
    Image.fromarray(image).save(path, "PNG")


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image data to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def build_foreground_mask(gray: np.ndarray, dark_delta: int, dilate: int) -> np.ndarray:
    """Create a text/content mask from a mostly light document image."""
    background = int(np.percentile(gray, 95))
    threshold = max(0, min(250, background - dark_delta))
    mask = gray < threshold

    if dilate > 0:
        kernel_size = (dilate * 2) + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return mask


def remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    """Remove tiny components before computing crop bounds."""
    if min_component_area <= 0:
        return mask

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == label] = True

    return cleaned


def remove_likely_edge_components(mask: np.ndarray) -> np.ndarray:
    """Drop long, thin components that look like scanner/page-edge borders."""
    height, width = mask.shape
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = mask.copy()
    for label in range(1, component_count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        touches_edge = x == 0 or y == 0 or x + w >= width or y + h >= height
        long_horizontal = w >= width * 0.50 and h <= max(12, height * 0.05)
        long_vertical = h >= height * 0.50 and w <= max(12, width * 0.05)

        if touches_edge and (long_horizontal or long_vertical):
            cleaned[labels == label] = False

    return cleaned


def crop_box_from_mask(
    mask: np.ndarray,
    padding: int,
    min_crop_ratio: float,
) -> CropBox | None:
    """Return padded crop bounds around foreground content."""
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return None

    height, width = mask.shape
    left = max(0, int(cols.min()) - padding)
    top = max(0, int(rows.min()) - padding)
    right = min(width, int(cols.max()) + padding + 1)
    bottom = min(height, int(rows.max()) + padding + 1)

    box = CropBox(left=left, top=top, right=right, bottom=bottom)
    crop_area = box.width * box.height
    image_area = width * height

    if image_area == 0 or crop_area / image_area < min_crop_ratio:
        return None

    return box


def detect_crop_box(image: np.ndarray, settings: CropSettings) -> CropBox | None:
    """Detect an OCR-safe crop box around likely Telugu text/content."""
    gray = grayscale(image)
    mask = build_foreground_mask(
        gray,
        dark_delta=settings.dark_delta,
        dilate=settings.dilate,
    )
    mask = remove_small_components(mask, settings.min_component_area)

    if settings.remove_edge_components:
        mask = remove_likely_edge_components(mask)

    return crop_box_from_mask(
        mask,
        padding=settings.padding,
        min_crop_ratio=settings.min_crop_ratio,
    )


def apply_crop(image: np.ndarray, box: CropBox | None) -> np.ndarray:
    """Apply the detected crop, then remove the remaining outer border."""
    cropped = image if box is None else image[
        box.top : box.bottom,
        box.left : box.right,
    ]
    height, width = cropped.shape[:2]

    if height <= FINAL_BORDER_CLIP * 2 or width <= FINAL_BORDER_CLIP * 2:
        raise ValueError(
            "Image is too small for the required "
            f"{FINAL_BORDER_CLIP}px border clip: {width}x{height}"
        )

    return cropped[
        FINAL_BORDER_CLIP : height - FINAL_BORDER_CLIP,
        FINAL_BORDER_CLIP : width - FINAL_BORDER_CLIP,
    ]


def test_output_path(path: Path) -> Path:
    """Return the side-by-side test output path for an image."""
    return path.with_name(f"{path.stem}.test{path.suffix}")


def output_path_for_input(path: Path, input_root: Path, output_dir: Path | None) -> Path:
    """Return either an output-dir mirror path or the original image path."""
    if output_dir is None:
        return path

    if input_root.is_file():
        return output_dir / path.name

    return output_dir / path.relative_to(input_root)


def process_png(
    path: Path,
    input_root: Path,
    output_dir: Path | None,
    test: bool,
    settings: CropSettings,
) -> bool:
    """Crop one PNG, overwriting it unless test/output mode is enabled."""
    output_path = test_output_path(path) if test else output_path_for_input(
        path,
        input_root=input_root,
        output_dir=output_dir,
    )

    try:
        image = load_png(path)
        box = detect_crop_box(image, settings)
        cropped = apply_crop(image, box)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_png(cropped, output_path)
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)
        return False

    if box is None:
        logger.info(
            "Processed %s -> %s "
            "(no content crop; clipped %spx border, output: %sx%s)",
            path,
            output_path,
            FINAL_BORDER_CLIP,
            cropped.shape[1],
            cropped.shape[0],
        )
    else:
        logger.info(
            "Processed %s -> %s "
            "(crop: %s,%s %sx%s; clipped %spx border, output: %sx%s)",
            path,
            output_path,
            box.left,
            box.top,
            box.width,
            box.height,
            FINAL_BORDER_CLIP,
            cropped.shape[1],
            cropped.shape[0],
        )
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
        description="Conservatively crop PNG images for Telugu OCR preprocessing."
    )
    parser.add_argument("input", help="PNG image path or directory to scan")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write cropped files to a mirrored output directory instead of overwriting",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Write name.test.png beside each source image instead of overwriting it",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Pixels to preserve around detected content",
    )
    parser.add_argument(
        "--dark-delta",
        type=int,
        default=45,
        help="Foreground must be this much darker than the estimated background",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=20_000,
        help="Ignore connected components smaller than this many pixels",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=4,
        help="Expand the foreground mask by this many pixels before cropping",
    )
    parser.add_argument(
        "--keep-edge-components",
        action="store_true",
        help="Keep long edge-touching components that may be page borders",
    )
    parser.add_argument(
        "--min-crop-ratio",
        type=float,
        default=0.01,
        help="Skip crops smaller than this fraction of the original image area",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        return 1

    if args.test and args.output_dir is not None:
        logger.error("--test and --output-dir cannot be used together")
        return 1

    pngs = find_pngs(input_path)
    if not pngs:
        logger.error("No PNG files found: %s", input_path)
        return 1

    settings = CropSettings(
        dark_delta=args.dark_delta,
        min_component_area=args.min_component_area,
        padding=args.padding,
        dilate=args.dilate,
        remove_edge_components=not args.keep_edge_components,
        min_crop_ratio=args.min_crop_ratio,
    )

    successful = sum(
        1
        for png in pngs
        if process_png(
            png,
            input_root=input_path,
            output_dir=args.output_dir,
            test=args.test,
            settings=settings,
        )
    )
    failed = len(pngs) - successful

    logger.info("Done: %s successful, %s failed", successful, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
