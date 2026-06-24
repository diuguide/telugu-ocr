#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
default_pages_root="$project_root/standardized_full/TeluguSeg/test/9"
configured_input_root="${PIPELINE_INPUT_ROOT:-}"
configured_output_root="${PIPELINE_OUTPUT_ROOT:-}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [OPTIONS] IMAGE_FILE
  $(basename "$0") [OPTIONS] DIRECTORY
  $(basename "$0") [OPTIONS] START_PAGE END_PAGE [PAGES_ROOT]

Process one raster image, recursively process a directory, or recursively
process numbered directories from START_PAGE through END_PAGE. Each result is
written beside its source as <name>.after.png. Intermediate files are kept in
a temporary directory.

Options:
  --output-dir DIRECTORY
      Write PNG outputs beneath DIRECTORY. Directory inputs retain their
      relative tree; page ranges retain their numbered page directories.
  --skip-standardize, --no-standardize
      Skip DPI/resolution, orientation, and RGB standardization. The source is
      copied unchanged into the temporary workspace before cropping.
  -h, --help
      Show this help message.

PAGES_ROOT defaults to:
  $default_pages_root

Example:
  $(basename "$0") writer/1/scan.jpg
  $(basename "$0") writer/
  $(basename "$0") 1 10
  $(basename "$0") --skip-standardize writer/1/scan.jpg
  $(basename "$0") writer/ --output-dir processed/
EOF
}

skip_standardize=false
cli_output_root=""
positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-standardize|--no-standardize)
            skip_standardize=true
            shift
            ;;
        --output-dir)
            if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
                echo "--output-dir requires a directory." >&2
                exit 2
            fi
            cli_output_root="$2"
            shift 2
            ;;
        --output-dir=*)
            cli_output_root="${1#*=}"
            if [[ -z "$cli_output_root" ]]; then
                echo "--output-dir requires a directory." >&2
                exit 2
            fi
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            positional+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done
set -- "${positional[@]}"

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
    input_path="$(realpath "$input_path")"

    mapfile -d '' sources < <(find_sources "$input_path" | sort -zV)
    if [[ -f "$input_path" ]]; then
        selected_input_root="$(dirname "$input_path")"
    else
        selected_input_root="$input_path"
    fi
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
    pages_root="$(realpath "$pages_root")"
    selected_input_root="$pages_root"

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

if [[ -n "$cli_output_root" && -n "$configured_output_root" ]]; then
    echo "Use either --output-dir or PIPELINE_OUTPUT_ROOT, not both." >&2
    exit 2
fi

if [[ -n "$cli_output_root" ]]; then
    cli_output_root="$(realpath -m "$cli_output_root")"
    if [[ "$cli_output_root" == "$selected_input_root" \
        || "$cli_output_root" == "$selected_input_root/"* ]]; then
        echo "--output-dir must be outside the selected input directory." >&2
        exit 2
    fi
    mkdir -p "$cli_output_root"
fi

if [[ -n "$configured_output_root" ]]; then
    if [[ -z "$configured_input_root" ]]; then
        echo "PIPELINE_INPUT_ROOT is required with PIPELINE_OUTPUT_ROOT." >&2
        exit 2
    fi
    configured_input_root="$(realpath "$configured_input_root")"
    configured_output_root="$(realpath -m "$configured_output_root")"
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

if [[ "$skip_standardize" == true ]]; then
    echo "Standardize images: skipped"
else
    echo "Standardize images: enabled"
fi
if [[ -n "$cli_output_root" ]]; then
    echo "Output directory: $cli_output_root"
elif [[ -n "$configured_output_root" ]]; then
    echo "Output directory: $configured_output_root"
else
    echo "Output directory: beside each source"
fi

for source in "${sources[@]}"; do
    source_name="$(basename "$source")"
    working="$work_dir/${source_name%.*}.png"

    if [[ -n "$cli_output_root" ]]; then
        if [[ "$source" != "$selected_input_root/"* ]]; then
            echo "Source is outside the selected input root: $source" >&2
            exit 1
        fi
        relative_source="${source#"$selected_input_root/"}"
        output="$cli_output_root/${relative_source%.*}.png"
        mkdir -p "$(dirname "$output")"
    elif [[ -n "$configured_output_root" ]]; then
        if [[ "$source" != "$configured_input_root/"* ]]; then
            echo "Source is outside PIPELINE_INPUT_ROOT: $source" >&2
            exit 1
        fi
        relative_source="${source#"$configured_input_root/"}"
        output="$configured_output_root/${relative_source%.*}.png"
        mkdir -p "$(dirname "$output")"
    else
        output="${source%.*}.after.png"
    fi

    staged_output="${output}.tmp.$$"

    if [[ "$skip_standardize" == true ]]; then
        cp -- "$source" "$working"
    else
        "$python_bin" "$script_dir/standardize_images.py" \
            "$source" \
            --output-dir "$work_dir"
    fi
    # First crop removes the scanner/page border once.
    "$python_bin" "$script_dir/crop_images.py" "$working"
    "$python_bin" "$script_dir/enhance_contrast_images.py" "$working"
    "$python_bin" "$script_dir/denoise_images.py" "$working"
    "$python_bin" "$script_dir/deskew_images.py" "$working"
    # The second crop tightens post-deskew bounds without clipping another 15px.
    "$python_bin" "$script_dir/crop_images.py" \
        "$working" \
        --skip-final-border-clip
    "$python_bin" "$script_dir/binarize_images.py" "$working"

    cp -- "$working" "$staged_output"
    mv -- "$staged_output" "$output"
    staged_output=""
    ((processed += 1))
    echo "Completed: $source -> $output"
done

echo "Done: $processed file(s) processed."
