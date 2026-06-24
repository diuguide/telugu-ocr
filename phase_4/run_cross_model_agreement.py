#!/usr/bin/env python3
"""Method C: compare two OCR results and create a human-review queue."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from statistics import mean, median

from phase4_validation_common import (
    COMPLETED_STATUSES,
    PROJECT_ROOT,
    image_key_sort,
    infer_input_kind,
    load_inventory,
    load_latest_results,
    natural_key,
    normalize,
    page_key_from_image,
    select_pages,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/evaluation/cross_model_agreement"
DEFAULT_FLUENCY_ROOT = PROJECT_ROOT / "data/evaluation/openai_fluency"
DEFAULT_ERROR_ROOT = PROJECT_ROOT / "data/evaluation/openai_error_detection"


def agreement_score(text_a: str, text_b: str) -> float:
    # SequenceMatcher considers two empty strings a perfect match. For OCR
    # validation, two failures provide no positive evidence of accuracy.
    if not text_a and not text_b:
        return 0.0
    return SequenceMatcher(None, text_a, text_b).ratio()


def usable_text(record: dict | None) -> str:
    if record and record.get("status") in COMPLETED_STATUSES:
        return normalize(record.get("text"))
    return ""


def record_status(record: dict | None) -> str:
    return str(record.get("status", "missing")) if record else "missing"


def review_reason(
    record_a: dict | None,
    record_b: dict | None,
    text_a: str,
    text_b: str,
    score: float,
    threshold: float,
    model_a: str,
    model_b: str,
) -> str:
    reasons = []
    if record_a is None or record_a.get("status") not in COMPLETED_STATUSES:
        reasons.append(f"{model_a}_missing_or_error")
    if record_b is None or record_b.get("status") not in COMPLETED_STATUSES:
        reasons.append(f"{model_b}_missing_or_error")
    if not text_a and not text_b:
        reasons.append("both_empty")
    elif not text_a:
        reasons.append(f"{model_a}_empty")
    elif not text_b:
        reasons.append(f"{model_b}_empty")
    if score < threshold:
        reasons.append("agreement_below_threshold")
    return ";".join(dict.fromkeys(reasons))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def pages_for_records(records: dict) -> set[str]:
    return {page_key_from_image(key) for key in records}


def keys_for_page(page_key: str, *collections: dict) -> list[tuple]:
    keys = {
        key
        for collection in collections
        for key in collection
        if page_key_from_image(key) == page_key
    }
    return sorted(keys, key=image_key_sort)


def load_page_results(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    latest = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok" and record.get("page_key"):
                latest[record["page_key"]] = record
    return latest


def optional_result_path(explicit: Path | None, default: Path) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return default if default.is_file() else None


def error_validation_metrics(record: dict | None) -> dict[str, int | float | str]:
    """Return comparable Method B rates while keeping absent joins visibly blank."""
    if not record:
        return {
            "detected_errors": "",
            "flagged_regions": "",
            "missing_words": "",
            "flagged_region_rate": "",
            "missing_word_rate": "",
            "combined_failure_rate": "",
        }
    expected = int(record.get("expected_words", 0) or 0)
    missing = int(record.get("missing_or_error_words", 0) or 0)
    flagged = int(record.get("flagged_region_count", 0) or 0)
    nonmissing = max(expected - missing, 0)
    return {
        "detected_errors": int(record.get("detected_error_count", 0) or 0),
        "flagged_regions": flagged,
        "missing_words": missing,
        "flagged_region_rate": flagged / nonmissing if nonmissing else 0,
        "missing_word_rate": missing / expected if expected else 0,
        "combined_failure_rate": (flagged + missing) / expected if expected else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_a", type=Path, help="first OCR results.jsonl")
    parser.add_argument("results_b", type=Path, help="second OCR results.jsonl")
    parser.add_argument(
        "--inventory",
        "--manifest",
        dest="inventory",
        type=Path,
        help="optional CSV of expected image paths; transcription is never read",
    )
    parser.add_argument("--model-a-name", help="default: first results parent folder")
    parser.add_argument("--model-b-name", help="default: second results parent folder")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-threshold", type=float, default=0.70)
    parser.add_argument("--limit-pages", type=int, help="quick deterministic test")
    parser.add_argument(
        "--sample-pages", type=int, help="balanced sample across split/writer strata"
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--page-keys", type=Path)
    parser.add_argument(
        "--complete-pages-only",
        action="store_true",
        help="compare only pages completed by both OCR models",
    )
    parser.add_argument(
        "--allow-mismatched-input-kinds",
        action="store_true",
        help="permit raw-versus-processed comparisons",
    )
    parser.add_argument("--fluency-a", type=Path)
    parser.add_argument("--fluency-b", type=Path)
    parser.add_argument("--errors-a", type=Path)
    parser.add_argument("--errors-b", type=Path)
    parser.add_argument(
        "--no-combined-report",
        action="store_true",
        help="skip joining available Method A and B outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.review_threshold <= 1:
        raise ValueError("--review-threshold must be between 0 and 1")

    results_a_path = args.results_a.expanduser().resolve()
    results_b_path = args.results_b.expanduser().resolve()
    inventory_path = args.inventory.expanduser().resolve() if args.inventory else None
    inventory = load_inventory(inventory_path)
    records_a, unmapped_a, malformed_a = load_latest_results(results_a_path)
    records_b, unmapped_b, malformed_b = load_latest_results(results_b_path)
    if not records_a or not records_b:
        raise RuntimeError("both result files must contain usable image paths")

    model_a = args.model_a_name or results_a_path.parent.name
    model_b = args.model_b_name or results_b_path.parent.name
    kind_a = infer_input_kind(records_a)
    kind_b = infer_input_kind(records_b)
    if kind_a != kind_b and not args.allow_mismatched_input_kinds:
        raise ValueError(
            f"input kinds differ ({model_a}={kind_a}, {model_b}={kind_b}); "
            "compare like with like or pass --allow-mismatched-input-kinds"
        )
    input_kind = kind_a if kind_a == kind_b else f"{kind_a}_vs_{kind_b}"

    pages_a = pages_for_records(records_a)
    pages_b = pages_for_records(records_b)
    common_page_keys = pages_a & pages_b
    unpaired_rows = [
        {
            "page_key": page_key,
            "present_in_model_a": str(page_key in pages_a).lower(),
            "present_in_model_b": str(page_key in pages_b).lower(),
            "human_review": "true",
            "review_reason": (
                f"missing_entire_page_from_{model_b}"
                if page_key in pages_a
                else f"missing_entire_page_from_{model_a}"
            ),
        }
        for page_key in sorted(pages_a ^ pages_b, key=natural_key)
    ]
    page_descriptors = []
    excluded_incomplete_rows = []
    for page_key in sorted(common_page_keys, key=natural_key):
        keys = keys_for_page(page_key, records_a, records_b, inventory)
        expected = len(keys)
        complete_a = sum(
            records_a.get(key, {}).get("status") in COMPLETED_STATUSES for key in keys
        )
        complete_b = sum(
            records_b.get(key, {}).get("status") in COMPLETED_STATUSES for key in keys
        )
        if args.complete_pages_only and (complete_a != expected or complete_b != expected):
            excluded_incomplete_rows.append(
                {
                    "page_key": page_key,
                    "model_a_completed_words": complete_a,
                    "model_b_completed_words": complete_b,
                    "expected_words": expected,
                    "human_review": "true",
                    "review_reason": "incomplete_page_excluded_by_complete_pages_only",
                }
            )
            continue
        split, writer_id, page_id, _ = keys[0]
        page_descriptors.append(
            {
                "page_key": page_key,
                "split": split,
                "writer_id": writer_id,
                "page_id": page_id,
                "completed_words": min(complete_a, complete_b),
                "expected_words": expected,
            }
        )
    pages = select_pages(
        page_descriptors,
        args.limit_pages,
        args.sample_pages,
        args.sample_seed,
        args.page_keys.expanduser().resolve() if args.page_keys else None,
    )
    if not pages:
        raise RuntimeError("the result files have no comparable pages")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT
        / f"{safe_name(model_a)}_vs_{safe_name(model_b)}"
        / safe_name(input_kind)
    )
    selected_page_keys = [page["page_key"] for page in pages]
    word_rows = []
    page_rows = []
    review_rows = []
    for page_key in selected_page_keys:
        page_word_rows = []
        page_text_a = []
        page_text_b = []
        for key in keys_for_page(page_key, records_a, records_b, inventory):
            record_a = records_a.get(key)
            record_b = records_b.get(key)
            text_a = usable_text(record_a)
            text_b = usable_text(record_b)
            score = agreement_score(text_a, text_b)
            reason = review_reason(
                record_a,
                record_b,
                text_a,
                text_b,
                score,
                args.review_threshold,
                model_a,
                model_b,
            )
            source_path = (
                (record_a or {}).get("input_path")
                or (record_b or {}).get("input_path")
                or inventory.get(key, {}).get("image_path", "")
            )
            row = {
                "input_kind": input_kind,
                "page_key": page_key,
                "image_index": key[3],
                "source_image_path": source_path,
                "model_a": model_a,
                "model_b": model_b,
                "model_a_status": record_status(record_a),
                "model_b_status": record_status(record_b),
                "model_a_text": text_a,
                "model_b_text": text_b,
                "agreement_score": f"{score:.6f}",
                "disagreement_score": f"{1 - score:.6f}",
                "human_review": str(bool(reason)).lower(),
                "review_reason": reason,
            }
            word_rows.append(row)
            page_word_rows.append(row)
            if reason:
                review_rows.append(row)
            page_text_a.append(text_a)
            page_text_b.append(text_b)

        joined_a = normalize(" ".join(page_text_a))
        joined_b = normalize(" ".join(page_text_b))
        page_score = agreement_score(joined_a, joined_b)
        flagged_regions = sum(row["human_review"] == "true" for row in page_word_rows)
        completed_a = sum(
            row["model_a_status"] in COMPLETED_STATUSES for row in page_word_rows
        )
        completed_b = sum(
            row["model_b_status"] in COMPLETED_STATUSES for row in page_word_rows
        )
        expected = len(page_word_rows)
        usable_a = sum(bool(row["model_a_text"]) for row in page_word_rows)
        usable_b = sum(bool(row["model_b_text"]) for row in page_word_rows)
        reasons = []
        if page_score < args.review_threshold:
            reasons.append("agreement_below_threshold")
        if flagged_regions:
            reasons.append("word_regions_flagged")
        if completed_a != expected or completed_b != expected:
            reasons.append("incomplete_page")
        page_rows.append(
            {
                "input_kind": input_kind,
                "page_key": page_key,
                "model_a": model_a,
                "model_b": model_b,
                "model_a_completed_words": completed_a,
                "model_b_completed_words": completed_b,
                "model_a_usable_words": usable_a,
                "model_b_usable_words": usable_b,
                "model_a_missing_word_rate": (
                    (expected - usable_a) / expected if expected else 0
                ),
                "model_b_missing_word_rate": (
                    (expected - usable_b) / expected if expected else 0
                ),
                "expected_words": expected,
                "model_a_text": joined_a,
                "model_b_text": joined_b,
                "agreement_score": f"{page_score:.6f}",
                "disagreement_score": f"{1 - page_score:.6f}",
                "flagged_word_regions": flagged_regions,
                "human_review": str(bool(reasons)).lower(),
                "review_reason": ";".join(reasons),
            }
        )

    write_csv(output_dir / "word_agreement.csv", list(word_rows[0]), word_rows)
    write_csv(output_dir / "page_agreement.csv", list(page_rows[0]), page_rows)
    write_csv(output_dir / "human_review.csv", list(word_rows[0]), review_rows)
    write_csv(
        output_dir / "unpaired_pages.csv",
        [
            "page_key",
            "present_in_model_a",
            "present_in_model_b",
            "human_review",
            "review_reason",
        ],
        unpaired_rows,
    )
    write_csv(
        output_dir / "excluded_incomplete_pages.csv",
        [
            "page_key",
            "model_a_completed_words",
            "model_b_completed_words",
            "expected_words",
            "human_review",
            "review_reason",
        ],
        excluded_incomplete_rows,
    )

    combined_rows = []
    method_paths = {}
    if not args.no_combined_report:
        method_paths = {
            "fluency_a": optional_result_path(
                args.fluency_a, DEFAULT_FLUENCY_ROOT / model_a / "results.jsonl"
            ),
            "fluency_b": optional_result_path(
                args.fluency_b, DEFAULT_FLUENCY_ROOT / model_b / "results.jsonl"
            ),
            "errors_a": optional_result_path(
                args.errors_a, DEFAULT_ERROR_ROOT / model_a / "results.jsonl"
            ),
            "errors_b": optional_result_path(
                args.errors_b, DEFAULT_ERROR_ROOT / model_b / "results.jsonl"
            ),
        }
        fluency_a = load_page_results(method_paths["fluency_a"])
        fluency_b = load_page_results(method_paths["fluency_b"])
        errors_a = load_page_results(method_paths["errors_a"])
        errors_b = load_page_results(method_paths["errors_b"])
        for page in page_rows:
            page_key = page["page_key"]
            error_a_metrics = error_validation_metrics(errors_a.get(page_key))
            error_b_metrics = error_validation_metrics(errors_b.get(page_key))
            combined_rows.append(
                {
                    **page,
                    "model_a_fluency_score": fluency_a.get(page_key, {}).get("score", ""),
                    "model_a_fluency_reason": fluency_a.get(page_key, {}).get("reason", ""),
                    "model_b_fluency_score": fluency_b.get(page_key, {}).get("score", ""),
                    "model_b_fluency_reason": fluency_b.get(page_key, {}).get("reason", ""),
                    "model_a_detected_errors": error_a_metrics["detected_errors"],
                    "model_a_flagged_regions": error_a_metrics["flagged_regions"],
                    "model_a_missing_words": error_a_metrics["missing_words"],
                    "model_a_flagged_region_rate": error_a_metrics[
                        "flagged_region_rate"
                    ],
                    "model_a_missing_word_rate": error_a_metrics[
                        "missing_word_rate"
                    ],
                    "model_a_combined_failure_rate": error_a_metrics[
                        "combined_failure_rate"
                    ],
                    "model_b_detected_errors": error_b_metrics["detected_errors"],
                    "model_b_flagged_regions": error_b_metrics["flagged_regions"],
                    "model_b_missing_words": error_b_metrics["missing_words"],
                    "model_b_flagged_region_rate": error_b_metrics[
                        "flagged_region_rate"
                    ],
                    "model_b_missing_word_rate": error_b_metrics[
                        "missing_word_rate"
                    ],
                    "model_b_combined_failure_rate": error_b_metrics[
                        "combined_failure_rate"
                    ],
                }
            )
        write_csv(
            output_dir / "combined_page_validation.csv",
            list(combined_rows[0]),
            combined_rows,
        )

    word_scores = [float(row["agreement_score"]) for row in word_rows]
    page_scores = [float(row["agreement_score"]) for row in page_rows]
    summary = {
        "method": "C_cross_model_agreement",
        "results_a": str(results_a_path),
        "results_b": str(results_b_path),
        "inventory": str(inventory_path) if inventory_path else None,
        "model_a": model_a,
        "model_b": model_b,
        "input_kind": input_kind,
        "review_threshold": args.review_threshold,
        "sampling": {
            "limit_pages": args.limit_pages,
            "sample_pages": args.sample_pages,
            "sample_seed": args.sample_seed if args.sample_pages else None,
            "page_keys_file": str(args.page_keys.resolve()) if args.page_keys else None,
        },
        "compared_pages": len(page_rows),
        "compared_word_regions": len(word_rows),
        "flagged_pages": sum(row["human_review"] == "true" for row in page_rows),
        "flagged_word_regions": len(review_rows),
        "unpaired_pages": len(unpaired_rows),
        "excluded_incomplete_pages": len(excluded_incomplete_rows),
        "total_pages_for_human_review": (
            sum(row["human_review"] == "true" for row in page_rows)
            + len(unpaired_rows)
            + len(excluded_incomplete_rows)
        ),
        "mean_word_agreement": mean(word_scores),
        "median_word_agreement": median(word_scores),
        "mean_page_agreement": mean(page_scores),
        "median_page_agreement": median(page_scores),
        "unmapped_a": unmapped_a,
        "unmapped_b": unmapped_b,
        "malformed_jsonl_lines_a": malformed_a,
        "malformed_jsonl_lines_b": malformed_b,
        "combined_report_written": bool(combined_rows),
        "method_result_paths": {
            key: str(value) if value else None for key, value in method_paths.items()
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Compared pages: {len(page_rows)}")
    print(f"Compared word regions: {len(word_rows)}")
    print(f"Flagged word regions: {len(review_rows)}")
    print(f"Mean page agreement: {summary['mean_page_agreement']:.6f}")
    print(f"Agreement reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
