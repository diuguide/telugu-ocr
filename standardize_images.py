#!/usr/bin/env python3
"""
Image Loading and Format Standardization Script

Converts all input images to a consistent format (PNG, 300 DPI minimum).
Upscales low-resolution images using super-resolution or interpolation.

Features:
- Support for multiple formats (JPG, PNG, BMP, GIF, TIFF, PDF)
- Automatic DPI standardization to 300 DPI minimum
- OpenCV INTER_CUBIC upscaling for low-resolution images
- Batch processing with progress tracking
"""

import sys
import argparse
import logging
import re
import os
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

# Optional imports
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def natural_sort_key(path: Path) -> list:
    """
    Generate a key for natural (numeric) sorting of file paths.
    
    This ensures that numeric components in paths are sorted numerically
    rather than lexicographically (e.g., 1, 2, 3, 10 instead of 1, 10, 2, 3).
    
    Args:
        path: A Path object to generate a sort key for
        
    Returns:
        A list containing alternating text and integer values for proper sorting
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(path))]


def resolve_teluguseg_input(input_path: Path) -> Path:
    """Resolve input path to TeluguSeg directory when dataset root is provided."""
    if input_path.is_dir() and (input_path / "TeluguSeg").is_dir():
        resolved = input_path / "TeluguSeg"
        logger.info(f"Detected dataset root. Using TeluguSeg input: {resolved}")
        return resolved
    return input_path


def is_teluguseg_layout(input_path: Path) -> bool:
    """Check whether the directory matches the expected TeluguSeg split layout."""
    if not input_path.is_dir():
        return False

    split_names = ("train", "val", "test")
    return any((input_path / split).is_dir() for split in split_names)


def _process_image_worker(
    input_path: str,
    output_path: str,
    output_format: str,
    target_dpi: int,
    min_width: int,
    upscale_factor: int,
) -> Tuple[bool, str, str]:
    """
    Worker function for parallel image processing.
    
    Must be module-level to be pickleable by ProcessPoolExecutor.
    
    Args:
        input_path: Path to input image
        output_path: Path to save standardized image
        output_format: Output format (PNG, JPG, BMP, TIFF)
        target_dpi: Target DPI
        min_width: Minimum width for upscaling
        upscale_factor: Upscaling factor
        
    Returns:
        Tuple of (success, input_path, output_path)
    """
    standardizer = ImageStandardizer(
        output_format=output_format,
        target_dpi=target_dpi,
        min_width=min_width,
        upscale_factor=upscale_factor,
    )
    success = standardizer.process_image(input_path, output_path)
    return (success, input_path, output_path)


class ImageStandardizer:
    """Standardizes images to consistent format, DPI, and resolution."""
    
    def __init__(
        self,
        output_format: str = "PNG",
        target_dpi: int = 300,
        min_width: int = 100,
        upscale_factor: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize ImageStandardizer.
        
        Args:
            output_format: Output format (PNG, JPG, BMP, TIFF)
            target_dpi: Target DPI for standardization (default 300)
            min_width: Minimum width in pixels (images smaller will be upscaled)
            upscale_factor: Upscaling factor (2, 3, or 4)
            max_workers: Number of parallel workers for ProcessPoolExecutor (default: min(cpu_count, 8))
        """
        self.output_format = output_format.upper()
        self.target_dpi = target_dpi
        self.min_width = min_width
        self.upscale_factor = upscale_factor or 2
        self.max_workers = max_workers or min(os.cpu_count() or 1, 8)
        logger.info("Using OpenCV INTER_CUBIC for upscaling")
        logger.info(f"Parallel processing with {self.max_workers} workers")
    
    def _load_image_from_pdf(self, pdf_path: str, page_num: int = 0) -> Optional[Image.Image]:
        """Load first page from PDF."""
        if not PDF_SUPPORT:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            return None
        
        try:
            pages = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
            if pages:
                return pages[0]
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
        return None
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image from file (supports multiple formats)."""
        try:
            file_ext = Path(image_path).suffix.lower()
            
            if file_ext == '.pdf':
                return self._load_image_from_pdf(image_path)
            else:
                img = Image.open(image_path)
                # Convert RGBA to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    return background
                elif img.mode != 'RGB':
                    return img.convert('RGB')
                return img
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _get_dpi_from_image(self, img: Image.Image) -> Tuple[int, int]:
        """Extract DPI from image metadata."""
        dpi = img.info.get('dpi', (72, 72))
        return dpi if isinstance(dpi, tuple) else (dpi, dpi)
    
    def _needs_upscaling(self, width: int) -> bool:
        """Check if image needs upscaling based on width."""
        return width < self.min_width
    
    def _upscale_opencv(self, cv_image: np.ndarray) -> np.ndarray:
        """Upscale image using OpenCV INTER_CUBIC interpolation."""
        height, width = cv_image.shape[:2]
        new_width = width * self.upscale_factor
        new_height = height * self.upscale_factor
        upscaled = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Upscaled {width}x{height} -> {new_width}x{new_height} using OpenCV INTER_CUBIC")
        return upscaled
    
    def _standardize_dpi(self, img: Image.Image) -> Image.Image:
        """Standardize image DPI to target DPI."""
        current_dpi = self._get_dpi_from_image(img)
        
        if current_dpi[0] < self.target_dpi or current_dpi[1] < self.target_dpi:
            # Rescale image to achieve target DPI
            scale_factor = self.target_dpi / min(current_dpi)
            new_size = (
                int(img.width * scale_factor),
                int(img.height * scale_factor)
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Standardized DPI from {current_dpi} to {self.target_dpi}")
        
        return img
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        Process a single image: load, standardize, and save.
        
        Args:
            input_path: Path to input image
            output_path: Path to save standardized image
            
        Returns:
            True if successful, False otherwise
        """
        # Load image
        img = self._load_image(input_path)
        if img is None:
            return False
        
        # Standardize DPI
        img = self._standardize_dpi(img)
        
        # Check if upscaling is needed
        if self._needs_upscaling(img.width):
            logger.info(f"Upscaling image {input_path} ({img.width}x{img.height}) -> "
                       f"({img.width * self.upscale_factor}x{img.height * self.upscale_factor})")
            
            # Convert to OpenCV format for upscaling
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            cv_image = self._upscale_opencv(cv_image)
            
            # Convert back to PIL
            img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Save with standardized DPI
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.output_format == 'JPG':
                img.save(output_path, 'JPEG', dpi=(self.target_dpi, self.target_dpi), quality=95)
            else:
                img.save(output_path, self.output_format, dpi=(self.target_dpi, self.target_dpi))
            
            logger.info(f"Saved standardized image to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {e}")
            return False
    
    def _copy_text_files(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
        relative_base: Optional[str] = None,
    ) -> int:
        """
        Copy all .txt files from input directory to output directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Copy from subdirectories recursively
            relative_base: Optional base directory to preserve in output paths
            
        Returns:
            Number of text files copied
        """
        input_path = Path(input_dir)
        relative_root = Path(relative_base) if relative_base is not None else input_path
        output_path = Path(output_dir)
        
        # Find all .txt files
        if recursive:
            text_files = list(input_path.rglob('*.txt'))
        else:
            text_files = list(input_path.glob('*.txt'))
        
        if not text_files:
            logger.info("No .txt files found to copy")
            return 0
        
        copied_count = 0
        for text_file in sorted(text_files, key=natural_sort_key):
            try:
                # Preserve directory structure
                rel_path = text_file.relative_to(relative_root)
                output_file = output_path / rel_path
                
                # Create parent directory if needed
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                with open(text_file, 'r', encoding='utf-8') as src:
                    content = src.read()
                with open(output_file, 'w', encoding='utf-8') as dst:
                    dst.write(content)
                
                logger.info(f"Copied text file to {output_file}")
                copied_count += 1
            except Exception as e:
                logger.error(f"Failed to copy text file {text_file}: {e}")
        
        return copied_count
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
        max_images: Optional[int] = None,
        relative_base: Optional[str] = None,
    ) -> dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for standardized images
            recursive: Process subdirectories recursively
            max_images: Optional cap on number of images to process
            relative_base: Optional base directory to preserve in output paths
            
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_dir)
        relative_root = Path(relative_base) if relative_base is not None else input_path

        # Keep the TeluguSeg top-level folder in the output tree when present.
        if relative_root.name == "TeluguSeg":
            relative_root = relative_root.parent
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all supported image files
        supported_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif')
        if PDF_SUPPORT:
            supported_extensions += ('*.pdf',)
        
        if recursive:
            image_files = []
            for ext in supported_extensions:
                image_files.extend(input_path.rglob(ext))
                image_files.extend(input_path.rglob(ext.upper()))
        else:
            image_files = []
            for ext in supported_extensions:
                image_files.extend(input_path.glob(ext))
                image_files.extend(input_path.glob(ext.upper()))
        
        # Sort using natural (numeric) sorting instead of lexicographic sorting
        image_files = sorted(set(image_files), key=natural_sort_key)
        discovered_total = len(image_files)
        if max_images is not None and max_images > 0:
            image_files = image_files[:max_images]

        stats = {
            'total': len(image_files),
            'discovered_total': discovered_total,
            'successful': 0,
            'failed': 0,
        }
        
        logger.info(
            f"Processing {stats['total']} images from {input_dir} "
            f"(discovered: {stats['discovered_total']})"
        )
        
        # Prepare list of tasks for parallel processing
        tasks = []
        for image_file in image_files:
            rel_path = image_file.relative_to(relative_root)
            output_file = output_path / rel_path.with_suffix(f'.{self.output_format.lower()}')
            tasks.append((str(image_file), str(output_file)))
        
        # Process images in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    _process_image_worker,
                    input_file,
                    output_file,
                    self.output_format,
                    self.target_dpi,
                    self.min_width,
                    self.upscale_factor,
                )
                for input_file, output_file in tasks
            ]
            
            # Collect results and update stats
            for i, future in enumerate(futures, 1):
                try:
                    success, input_file, output_file = future.result()
                    if success:
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                    
                    # Log progress
                    if i % max(1, len(futures) // 10) == 0 or i == len(futures):
                        logger.info(f"Progress: {i}/{stats['total']} images processed")
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    stats['failed'] += 1
        
        # Copy text files after image processing
        text_files_copied = self._copy_text_files(
            str(input_path),
            str(output_path),
            recursive,
            str(relative_root),
        )
        stats['text_files_copied'] = text_files_copied
        
        return stats


def main():
    """Command-line interface for image standardization."""
    parser = argparse.ArgumentParser(
        description="Standardize images to consistent format, DPI, and resolution"
    )
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('output', help='Output image or directory')
    parser.add_argument(
        '--format',
        default='PNG',
        choices=['PNG', 'JPG', 'BMP', 'TIFF'],
        help='Output image format (default: PNG)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Target DPI (default: 300)'
    )
    parser.add_argument(
        '--min-width',
        type=int,
        default=100,
        help='Minimum width in pixels for upscaling trigger (default: 100)'
    )
    parser.add_argument(
        '--upscale-factor',
        type=int,
        choices=[2, 3, 4],
        default=2,
        help='Upscaling factor for OpenCV interpolation (default: 2)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process directories recursively'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Process at most N images (useful for runtime benchmarking)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: min(cpu_count, 8))'
    )
    
    args = parser.parse_args()
    
    # Initialize standardizer
    standardizer = ImageStandardizer(
        output_format=args.format,
        target_dpi=args.dpi,
        min_width=args.min_width,
        upscale_factor=args.upscale_factor,
        max_workers=args.max_workers,
    )
    
    original_input_path = Path(args.input)
    input_path = resolve_teluguseg_input(original_input_path)
    recursive = args.recursive

    if input_path.is_dir() and is_teluguseg_layout(input_path) and not recursive:
        recursive = True
        logger.info("Detected TeluguSeg directory layout. Enabling recursive processing automatically.")
    
    # Process single file or directory
    if input_path.is_file():
        logger.info(f"Processing single image: {input_path}")
        success = standardizer.process_image(str(input_path), args.output)
        sys.exit(0 if success else 1)
    elif input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        stats = standardizer.process_directory(
            str(input_path),
            args.output,
            recursive,
            args.max_images,
            str(original_input_path),
        )
        logger.info(
            f"Processing complete: {stats['successful']} successful, "
            f"{stats['failed']} failed (processed: {stats['total']}, discovered: {stats['discovered_total']}, "
            f"text files copied: {stats['text_files_copied']})"
        )
        sys.exit(0 if stats['failed'] == 0 else 1)
    else:
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
