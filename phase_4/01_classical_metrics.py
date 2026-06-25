#!/usr/bin/env python3
"""Calculate CER/WER for one OCR result or every deliverable OCR result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import unicodedata

from jiwer import cer, wer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "bin/phase_1/ground_truth/manifests/ground_truth_manifest.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "ground_truth/results.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/phase_4/classical"
DEFAULT_RESULTS_ROOT = (
    REPOSITORY_ROOT / "data/processed_ocr_responses"
)
DEFAULT_BATCH_OUTPUT = (
    REPOSITORY_ROOT / "data/llm_validations/CER-WER-validation"
)
DEFAULT_REPORT_NAME = "cer_wer_metrics.txt"
COMPLETED_STATUSES = {"ok", "empty_output"}


def normalize(text):
    return unicodedata.normalize("NFC", " ".join((text or "").split()))


def image_key(path):
    """Map raw and processed paths to the same split/writer/page/image key."""
    parts = Path(path).parts

    # OCR result writers record this canonical path separately from the absolute
    # input path (for example, ``test/9/13/1.png``).  Accepting it keeps metric
    # matching independent of the name chosen for a preprocessing directory.
    if len(parts) == 4:
        return tuple(Path(*parts).with_suffix("").parts)

    for directory in ("processed_sample_set", "sample_set"):
        if directory in parts:
            relative = Path(*parts[parts.index(directory) + 1 :]).with_suffix("")
            if len(relative.parts) == 4:
                return tuple(relative.parts)
    return None


def load_manifest(path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            key = image_key(row["image_path"])
            if key is None:
                raise ValueError(f"invalid image path in manifest: {row['image_path']}")
            row["ground_truth_text"] = normalize(row["ground_truth_text"])
            rows[key] = row
    return rows


def load_completed_results(path, manifest):
    """Read a JSONL snapshot and retain the latest completed record per image."""
    completed = {}
    error_records = 0
    unmapped_records = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping incomplete line {line_number} in {path}")
                continue

            if record.get("status") not in COMPLETED_STATUSES:
                error_records += 1
                continue
            key = image_key(record.get("relative_input_path", ""))
            if key is None:
                key = image_key(record.get("input_path", ""))
            if key not in manifest:
                unmapped_records += 1
                continue
            completed[key] = record

    return completed, error_records, unmapped_records


def image_sort_key(item):
    key, _ = item
    split, writer, page, image = key
    first_number = image.split("_")[0]
    return split, int(writer), int(page), int(first_number), image


def calculate_metrics(completed, manifest):
    word_rows = []
    page_members = {}

    for key, record in sorted(completed.items(), key=image_sort_key):
        ground_truth = manifest[key]
        reference = ground_truth["ground_truth_text"]
        hypothesis = normalize(record.get("text", ""))
        word_rows.append(
            {
                "page_key": ground_truth["page_key"],
                "image_index": ground_truth["image_index"],
                "source_image_path": ground_truth["image_path"],
                "input_image_path": record["input_path"],
                "ground_truth_text": reference,
                "prediction_text": hypothesis,
                "cer": f"{cer(reference, hypothesis):.6f}",
                "wer": f"{wer(reference, hypothesis):.6f}",
                "exact_match": str(reference == hypothesis).lower(),
                "status": record["status"],
                "suspicious_output": str(
                    record.get("suspicious_output", False)
                ).lower(),
            }
        )
        page_members.setdefault(ground_truth["page_key"], []).append(
            (ground_truth, hypothesis)
        )

    expected_page_sizes = {}
    for row in manifest.values():
        expected_page_sizes[row["page_key"]] = expected_page_sizes.get(row["page_key"], 0) + 1

    page_rows = []
    for page_key, members in page_members.items():
        reference = normalize(" ".join(row["ground_truth_text"] for row, _ in members))
        hypothesis = normalize(" ".join(text for _, text in members))
        first = members[0][0]
        completed_words = len(members)
        expected_words = expected_page_sizes[page_key]
        page_rows.append(
            {
                "page_key": page_key,
                "split": first["split"],
                "writer_id": first["writer_id"],
                "page_id": first["page_id"],
                "completed_words": completed_words,
                "expected_words": expected_words,
                "complete_page": str(completed_words == expected_words).lower(),
                "ground_truth_text": reference,
                "prediction_text": hypothesis,
                "cer": f"{cer(reference, hypothesis):.6f}",
                "wer": f"{wer(reference, hypothesis):.6f}",
            }
        )

    references = [row["ground_truth_text"] for row in word_rows]
    hypotheses = [row["prediction_text"] for row in word_rows]
    exact_matches = sum(reference == hypothesis for reference, hypothesis in zip(references, hypotheses))
    summary = {
        "manifest_records": len(manifest),
        "completed_records": len(word_rows),
        "coverage": f"{len(word_rows) / len(manifest):.6f}",
        "completed_pages": sum(row["complete_page"] == "true" for row in page_rows),
        "partial_pages": sum(row["complete_page"] == "false" for row in page_rows),
        "empty_outputs": sum(not hypothesis for hypothesis in hypotheses),
        "suspicious_outputs": sum(
            row["suspicious_output"] == "true" for row in word_rows
        ),
        "exact_matches": exact_matches,
        "exact_accuracy": f"{exact_matches / len(word_rows):.6f}",
        "mean_word_cer": f"{sum(float(row['cer']) for row in word_rows) / len(word_rows):.6f}",
        "mean_word_wer": f"{sum(float(row['wer']) for row in word_rows) / len(word_rows):.6f}",
        "mean_page_cer": f"{sum(float(row['cer']) for row in page_rows) / len(page_rows):.6f}",
        "mean_page_wer": f"{sum(float(row['wer']) for row in page_rows) / len(page_rows):.6f}",
        "corpus_cer": f"{cer(references, hypotheses):.6f}",
        "corpus_wer": f"{wer(references, hypotheses):.6f}",
    }
    return word_rows, page_rows, summary


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "results.jsonl or directory searched recursively for results.jsonl "
            f"(default: {DEFAULT_RESULTS_ROOT})"
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="single-result CSV directory or batch text-report directory",
    )
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    return parser.parse_args()


def configuration_fields(results_path, results_root):
    relative = results_path.relative_to(results_root)
    parts = relative.parent.parts
    model = parts[0] if parts else "unknown"
    input_kind = parts[1] if len(parts) > 1 else "unknown"
    run = parts[2] if len(parts) > 2 else input_kind
    return {
        "configuration": "/".join(parts) or results_path.parent.name,
        "model": model,
        "input_kind": input_kind,
        "run": run,
    }


def evaluate_result(results_path, manifest_path, manifest):
    completed, error_records, unmapped_records = load_completed_results(
        results_path, manifest
    )
    if not completed:
        raise RuntimeError("no completed records matched the ground-truth manifest")
    word_rows, page_rows, summary = calculate_metrics(completed, manifest)
    summary.update(
        {
            "results_path": str(results_path),
            "manifest_path": str(manifest_path),
            "error_records_skipped": error_records,
            "unmapped_records_skipped": unmapped_records,
        }
    )
    return word_rows, page_rows, summary


def write_batch_report(path, results_root, manifest_path, rows, failures):
    lines = [
        "CER/WER VALIDATION REPORT",
        f"Results root: {results_root}",
        f"Ground-truth manifest: {manifest_path}",
        f"Configurations evaluated: {len(rows)}",
        "",
    ]
    for number, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"CONFIGURATION {number}: {row['configuration']}",
                f"Input path: {row['results_path']}",
                f"Model: {row['model']}",
                f"Input kind: {row['input_kind']}",
                f"Run: {row['run']}",
                f"CER: {row['corpus_cer']}",
                f"WER: {row['corpus_wer']}",
                f"Coverage: {row['completed_records']}/{row['manifest_records']} "
                f"({row['coverage']})",
                f"Complete pages: {row['completed_pages']}",
                f"Partial pages: {row['partial_pages']}",
                f"Empty outputs: {row['empty_outputs']}",
                f"Exact matches: {row['exact_matches']}",
                f"Exact accuracy: {row['exact_accuracy']}",
                f"Mean word CER: {row['mean_word_cer']}",
                f"Mean word WER: {row['mean_word_wer']}",
                f"Mean page CER: {row['mean_page_cer']}",
                f"Mean page WER: {row['mean_page_wer']}",
                f"Error records skipped: {row['error_records_skipped']}",
                f"Unmapped records skipped: {row['unmapped_records_skipped']}",
                "",
            ]
        )
    if failures:
        lines.append("FAILURES")
        for name, error in failures:
            lines.extend([f"Configuration: {name}", f"Error: {error}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_batch(results_root, manifest_path, output_dir, report_name):
    result_paths = sorted(results_root.rglob("results.jsonl"))
    if not result_paths:
        raise FileNotFoundError(f"no results.jsonl files found under {results_root}")
    manifest = load_manifest(manifest_path)
    rows = []
    failures = []
    for results_path in result_paths:
        identity = configuration_fields(results_path, results_root)
        try:
            _, _, summary = evaluate_result(results_path, manifest_path, manifest)
        except Exception as exc:
            failures.append((identity["configuration"], str(exc)))
            continue
        rows.append({**identity, **summary})
        print(
            f"{identity['configuration']}: CER={summary['corpus_cer']} "
            f"WER={summary['corpus_wer']} coverage={summary['coverage']}"
        )
    report_path = output_dir / report_name
    write_batch_report(
        report_path, results_root, manifest_path, rows, failures
    )
    print(f"Consolidated metrics written to: {report_path}")
    return 1 if failures else 0


def main():
    args = parse_args()
    results_path = args.results.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if results_path.is_dir():
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else DEFAULT_BATCH_OUTPUT
        )
        return run_batch(
            results_path, manifest_path, output_dir, args.report_name
        )
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT
    )

    manifest = load_manifest(manifest_path)
    word_rows, page_rows, summary = evaluate_result(
        results_path, manifest_path, manifest
    )
    write_csv(output_dir / "word_metrics.csv", word_rows, list(word_rows[0]))
    write_csv(output_dir / "page_metrics.csv", page_rows, list(page_rows[0]))
    write_csv(output_dir / "summary.csv", [summary], list(summary))

    print(
        f"Scored {len(word_rows)}/{len(manifest)} completed records "
        f"({summary['coverage']} coverage)."
    )
    print(f"CER: {summary['corpus_cer']}  WER: {summary['corpus_wer']}")
    print(f"Metrics written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
