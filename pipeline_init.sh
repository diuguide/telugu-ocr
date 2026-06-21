#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_pages_root="$project_root/standardized_full/TeluguSeg/test/9"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") IMAGE_FILE
  $(basename "$0") DIRECTORY
  $(basename "$0") START_PAGE END_PAGE [PAGES_ROOT]

Process one raster image, recursively process a directory, or recursively
process numbered directories from START_PAGE through END_PAGE. Each result is
written beside its source as <name>.after.png. Intermediate files are kept in
a temporary directory.

PAGES_ROOT defaults to:
  $default_pages_root

Example:
  $(basename "$0") writer/1/scan.jpg
  $(basename "$0") writer/
  $(basename "$0") 1 10
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

find_sources() {
    find "$@" -type f \
        \( -iname '*.bmp' \
        -o -iname '*.gif' \
        -o -iname '*.jpeg' \
        -o -iname '*.jpg' \
        -o -iname '*.png' \
        -o -iname '*.tif' \
        -o -iname '*.tiff' \
        -o -iname '*.webp' \) \
        ! -iname '*.after.*' \
        ! -iname '*.test*.*' \
        -print0
}

sources=()

if [[ $# -eq 1 ]]; then
    input_path="$1"

    if [[ ! -e "$input_path" ]]; then
        echo "Input path not found: $input_path" >&2
        exit 1
    fi

    mapfile -d '' sources < <(find_sources "$input_path" | sort -zV)
else
    start_page="$1"
    end_page="$2"
    pages_root="${3:-$default_pages_root}"

    if [[ ! "$start_page" =~ ^[0-9]+$ || ! "$end_page" =~ ^[0-9]+$ ]]; then
        echo "START_PAGE and END_PAGE must be non-negative integers." >&2
        exit 2
    fi

    if (( start_page > end_page )); then
        echo "START_PAGE must be less than or equal to END_PAGE." >&2
        exit 2
    fi

    if [[ ! -d "$pages_root" ]]; then
        echo "Pages root not found: $pages_root" >&2
        exit 1
    fi

    page_dirs=()
    for ((page = start_page; page <= end_page; page++)); do
        page_dir="$pages_root/$page"
        if [[ ! -d "$page_dir" ]]; then
            echo "Page directory not found: $page_dir" >&2
            exit 1
        fi
        page_dirs+=("$page_dir")
    done

    mapfile -d '' sources < <(find_sources "${page_dirs[@]}" | sort -zV)
fi

if [[ ${#sources[@]} -eq 0 ]]; then
    echo "No supported raster images found." >&2
    exit 1
fi

if [[ -x "$project_root/.venv/bin/python" ]]; then
    python_bin="$project_root/.venv/bin/python"
else
    python_bin="python3"
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/telugu-image-pipeline.XXXXXX")"
staged_output=""

cleanup() {
    rm -rf "$work_dir"
    if [[ -n "$staged_output" ]]; then
        rm -f -- "$staged_output"
    fi
}

trap cleanup EXIT

processed=0

for source in "${sources[@]}"; do
    source_name="$(basename "$source")"
    working="$work_dir/${source_name%.*}.png"
    output="${source%.*}.after.png"
    staged_output="${output}.tmp.$$"

    "$python_bin" "$project_root/bin/standardize_images.py" \
        "$source" \
        --output-dir "$work_dir"
    "$python_bin" "$project_root/bin/crop_images.py" "$working"
    "$python_bin" "$project_root/bin/enhance_contrast_images.py" "$working"
    "$python_bin" "$project_root/bin/denoise_images.py" "$working"
    "$python_bin" "$project_root/bin/deskew_images.py" "$working"
    "$python_bin" "$project_root/bin/crop_images.py" "$working"
    "$python_bin" "$project_root/bin/binarize_images.py" "$working"

    cp -- "$working" "$staged_output"
    mv -- "$staged_output" "$output"
    staged_output=""
    ((processed += 1))
    echo "Completed: $source -> $output"
done

echo "Done: $processed file(s) processed."
