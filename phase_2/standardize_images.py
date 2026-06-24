#!/usr/bin/env python3
"""Standardize raster images as RGB PNG files for OCR preprocessing."""

import argparse
import logging
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


def natural_sort_key(path: Path) -> list[str | int]:
    """Return a path sort key that orders numeric names naturally."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def source_dpi(image: Image.Image) -> tuple[float, float] | None:
    """Return valid embedded DPI metadata, or None when it is unavailable."""
    dpi = image.info.get("dpi")
    if dpi is None:
        return None

    values = dpi if isinstance(dpi, tuple) else (dpi, dpi)
    if len(values) < 2:
        return None

    try:
        horizontal, vertical = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return None

    if horizontal <= 0 or vertical <= 0:
        return None

    return horizontal, vertical


def load_image(path: Path) -> tuple[Image.Image, tuple[float, float] | None]:
    """Load the first raster frame as oriented RGB on a white background."""
    with Image.open(path) as source:
        source.seek(0)
        dpi = source_dpi(source)
        image = ImageOps.exif_transpose(source)

        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            foreground = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, foreground).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        else:
            image = image.copy()

    return image, dpi


def standardize_resolution(
    image: Image.Image,
    dpi: tuple[float, float] | None,
    target_dpi: int,
    min_width: int,
) -> Image.Image:
    """Upscale for explicit low DPI or insufficient pixel width."""
    scale = 1.0

    if dpi is not None and min(dpi) < target_dpi:
        scale = target_dpi / min(dpi)

    if image.width * scale < min_width:
        scale = max(scale, min_width / image.width)

    if scale <= 1.0:
        return image

    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def save_png(image: Image.Image, path: Path, target_dpi: int) -> None:
    """Save an RGB image as PNG with standardized DPI metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", dpi=(target_dpi, target_dpi))


def test_output_path(path: Path) -> Path:
    """Return the side-by-side test output path for an image."""
    return path.with_name(f"{path.stem}.test.png")


def output_path_for_input(
    path: Path,
    input_root: Path,
    output_dir: Path | None,
    test: bool,
) -> Path:
    """Return the PNG output path for a source image."""
    if test:
        return test_output_path(path)

    if output_dir is None:
        return path.with_suffix(".png")

    relative_path = path.name if input_root.is_file() else path.relative_to(input_root)
    return (output_dir / relative_path).with_suffix(".png")


def process_image(
    path: Path,
    output_path: Path,
    target_dpi: int,
    min_width: int,
) -> bool:
    """Standardize one raster image and write it as PNG."""
    try:
        image, dpi = load_image(path)
        original_size = image.size
        standardized = standardize_resolution(
            image,
            dpi=dpi,
            target_dpi=target_dpi,
            min_width=min_width,
        )
        save_png(standardized, output_path, target_dpi=target_dpi)
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)
        return False

    logger.info(
        "Processed %s -> %s (%sx%s -> %sx%s, source DPI: %s)",
        path,
        output_path,
        original_size[0],
        original_size[1],
        standardized.width,
        standardized.height,
        "unknown" if dpi is None else f"{dpi[0]:.1f}x{dpi[1]:.1f}",
    )
    return True


def find_images(path: Path) -> list[Path]:
    """Find supported raster images from one file or a recursive directory."""
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else []

    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            and not item.stem.lower().endswith((".after", ".test"))
        ),
        key=natural_sort_key,
    )


def find_output_collision(
    images: list[Path],
    input_root: Path,
    output_dir: Path | None,
    test: bool,
) -> tuple[Path, Path, Path] | None:
    """Return the first pair of sources that resolve to the same output."""
    outputs: dict[Path, Path] = {}
    for image in images:
        output = output_path_for_input(image, input_root, output_dir, test)
        previous = outputs.get(output)
        if previous is not None and previous != image:
            return previous, image, output
        outputs[output] = image
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively standardize a raster image or directory as RGB PNG "
            "for OCR preprocessing."
        )
    )
    parser.add_argument("input", help="Image file or directory to scan recursively")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write PNG files to a mirrored directory instead of beside the inputs",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Write name.test.png beside each source image",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI and minimum explicit source DPI (default: 300)",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=100,
        help="Upscale images narrower than this pixel width (default: 100)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if args.test and args.output_dir is not None:
        logger.error("--test and --output-dir cannot be used together")
        return 1

    if args.dpi <= 0:
        logger.error("DPI must be positive: %s", args.dpi)
        return 1

    if args.min_width <= 0:
        logger.error("Minimum width must be positive: %s", args.min_width)
        return 1

    if not input_path.exists():
        logger.error("Input path not found: %s", input_path)
        return 1

    images = find_images(input_path)
    if not images:
        logger.error("No supported raster images found: %s", input_path)
        return 1

    collision = find_output_collision(
        images,
        input_root=input_path,
        output_dir=args.output_dir,
        test=args.test,
    )
    if collision is not None:
        first, second, output = collision
        logger.error(
            "Output collision: %s and %s would both write %s",
            first,
            second,
            output,
        )
        return 1

    successful = sum(
        1
        for image in images
        if process_image(
            image,
            output_path=output_path_for_input(
                image,
                input_root=input_path,
                output_dir=args.output_dir,
                test=args.test,
            ),
            target_dpi=args.dpi,
            min_width=args.min_width,
        )
    )
    failed = len(images) - successful

    logger.info("Done: %s successful, %s failed", successful, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
