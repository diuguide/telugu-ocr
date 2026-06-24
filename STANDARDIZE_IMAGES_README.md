# Image standardization

`standardize_images.py` converts raster images to oriented RGB PNG files for
the OCR preprocessing pipeline.

Supported inputs: BMP, GIF, JPEG, PNG, TIFF, and WebP.

## Behavior

- A single file is processed directly.
- A directory is always scanned recursively, including all nested page folders.
- Without an output option, PNG output is written beside each source. Existing
  PNG input is overwritten; other formats retain their original file and gain a
  same-stem PNG.
- `--output-dir` mirrors a directory input's relative folder structure.
- `--test` writes `name.test.png` beside each source.
- EXIF orientation is applied and transparency is flattened onto white.
- Explicit source DPI below the target is upscaled. Images without DPI metadata
  are not enlarged solely because metadata is missing.
- Images below `--min-width` are enlarged to that width while preserving aspect
  ratio.

Directory scans ignore existing `.after` and `.test` images.

## Usage

Standardize one file beside its source:

```bash
python3 bin/phase_2/standardize_images.py writer/1/scan.jpg
```

Recursively standardize an entire writer directory in place:

```bash
python3 bin/phase_2/standardize_images.py writer/
```

Recursively mirror standardized files into another directory:

```bash
python3 bin/phase_2/standardize_images.py writer/ --output-dir standardized_writer/
```

Write a side-by-side test result:

```bash
python3 bin/phase_2/standardize_images.py writer/1/scan.jpg --test
```

Use custom resolution settings:

```bash
python3 bin/phase_2/standardize_images.py writer/ --dpi 300 --min-width 100
```

## Pipeline integration

`pipeline_init.sh` runs standardization first, followed by:

```text
initial crop -> contrast -> denoise -> deskew -> final crop -> binarize
```

Process one file, one recursive directory, or page directories 1 through 10:

```bash
bin/phase_2/pipeline_init.sh writer/1/scan.jpg
bin/phase_2/pipeline_init.sh writer/
bin/phase_2/pipeline_init.sh 1 10
```
