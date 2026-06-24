#!/usr/bin/env python3
"""Compare raw OCR, preprocessing run 1, and preprocessing run 2."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from common import (
    bootstrap_mean_ci, common_parser, expected_page_sizes, load_csv, load_manifest,
    load_results, markdown_table, mean, normalize, page_metric_rows, paired_test,
    resolve_args, summarize_metrics, svg_bar, svg_scatter, text_metrics,
    word_metric_rows, write_csv, write_json,
)


MODELS = ("surya", "tesseract")
STAGES = ("raw", "run_1", "run_2")
COMPARISONS = (
    ("raw_to_run_1", "raw", "run_1"),
    ("run_1_to_run_2", "run_1", "run_2"),
    ("raw_to_run_2", "raw", "run_2"),
)


def stage_path(data_root: Path, model: str, stage: str) -> Path:
    root = data_root / "processed_ocr_responses" / model
    if stage == "raw":
        return root / "raw/results.jsonl"
    return root / "processed" / stage / "results.jsonl"


def load_stages(data_root: Path, manifest: dict) -> tuple[dict, dict]:
    stages, counts = {}, {}
    for model in MODELS:
        for stage in STAGES:
            path = stage_path(data_root, model, stage)
            if not path.is_file():
                raise FileNotFoundError(f"required OCR result is missing: {path}")
            stages[(model, stage)], counts[(model, stage)] = load_results(path, manifest)
    return stages, counts


def summarize_delta(
    model: str,
    comparison: str,
    baseline_stage: str,
    candidate_stage: str,
    level: str,
    metric: str,
    rows: list[dict],
    seed: int,
    lower_is_better: bool,
) -> dict:
    deltas = [float(row[f"{metric}_delta"]) for row in rows]
    ci_low, ci_high = bootstrap_mean_ci(deltas, seed)
    test = paired_test(deltas)
    wins = sum(value < 0 if lower_is_better else value > 0 for value in deltas)
    losses = sum(value > 0 if lower_is_better else value < 0 for value in deltas)
    ties = len(deltas) - wins - losses
    baseline_mean = mean(float(row[f"baseline_{metric}"]) for row in rows)
    candidate_mean = mean(float(row[f"candidate_{metric}"]) for row in rows)
    return {
        "model": model,
        "comparison": comparison,
        "baseline_stage": baseline_stage,
        "candidate_stage": candidate_stage,
        "level": level,
        "metric": metric,
        "pairs": len(rows),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "absolute_change": candidate_mean - baseline_mean,
        "relative_change": (
            (candidate_mean - baseline_mean) / baseline_mean
            if baseline_mean else float("nan")
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        **test,
    }


def build_word_pair(
    model: str,
    comparison: str,
    baseline_stage: str,
    candidate_stage: str,
    key: tuple[str, str, str, str],
    reference: str,
    page_key: str,
    baseline: dict,
    candidate: dict,
) -> dict:
    baseline_metric = text_metrics(reference, baseline["text"])
    candidate_metric = text_metrics(reference, candidate["text"])
    baseline_missing = float(not baseline["text"])
    candidate_missing = float(not candidate["text"])
    baseline_runtime = float(baseline.get("elapsed_seconds") or 0)
    candidate_runtime = float(candidate.get("elapsed_seconds") or 0)
    return {
        "model": model,
        "comparison": comparison,
        "baseline_stage": baseline_stage,
        "candidate_stage": candidate_stage,
        "image_key": "/".join(key),
        "page_key": page_key,
        "reference": reference,
        "baseline_hypothesis": baseline["text"],
        "candidate_hypothesis": candidate["text"],
        "baseline_cer": baseline_metric["cer"],
        "candidate_cer": candidate_metric["cer"],
        "cer_delta": candidate_metric["cer"] - baseline_metric["cer"],
        "baseline_wer": baseline_metric["wer"],
        "candidate_wer": candidate_metric["wer"],
        "wer_delta": candidate_metric["wer"] - baseline_metric["wer"],
        "baseline_exact_match": float(baseline_metric["exact_match"]),
        "candidate_exact_match": float(candidate_metric["exact_match"]),
        "exact_match_delta": (
            float(candidate_metric["exact_match"])
            - float(baseline_metric["exact_match"])
        ),
        "baseline_missing_output": baseline_missing,
        "candidate_missing_output": candidate_missing,
        "missing_output_delta": candidate_missing - baseline_missing,
        "baseline_runtime": baseline_runtime,
        "candidate_runtime": candidate_runtime,
        "runtime_delta": candidate_runtime - baseline_runtime,
    }


def build_page_pair(model: str, comparison: str, members: list[dict]) -> dict:
    members = sorted(
        members,
        key=lambda row: int(
            row["image_key"].rsplit("/", 1)[-1].split("_", 1)[0]
        ),
    )
    baseline_stage = members[0]["baseline_stage"]
    candidate_stage = members[0]["candidate_stage"]
    reference = normalize(" ".join(row["reference"] for row in members))
    baseline_text = normalize(
        " ".join(row["baseline_hypothesis"] for row in members)
    )
    candidate_text = normalize(
        " ".join(row["candidate_hypothesis"] for row in members)
    )
    baseline_metric = text_metrics(reference, baseline_text)
    candidate_metric = text_metrics(reference, candidate_text)
    return {
        "model": model,
        "comparison": comparison,
        "baseline_stage": baseline_stage,
        "candidate_stage": candidate_stage,
        "page_key": members[0]["page_key"],
        "paired_words": len(members),
        "baseline_cer": baseline_metric["cer"],
        "candidate_cer": candidate_metric["cer"],
        "cer_delta": candidate_metric["cer"] - baseline_metric["cer"],
        "baseline_wer": baseline_metric["wer"],
        "candidate_wer": candidate_metric["wer"],
        "wer_delta": candidate_metric["wer"] - baseline_metric["wer"],
        "baseline_exact_match": float(baseline_metric["exact_match"]),
        "candidate_exact_match": float(candidate_metric["exact_match"]),
        "exact_match_delta": (
            float(candidate_metric["exact_match"])
            - float(baseline_metric["exact_match"])
        ),
        "baseline_missing_output": mean(
            row["baseline_missing_output"] for row in members
        ),
        "candidate_missing_output": mean(
            row["candidate_missing_output"] for row in members
        ),
        "missing_output_delta": mean(
            row["missing_output_delta"] for row in members
        ),
        "baseline_runtime": sum(row["baseline_runtime"] for row in members),
        "candidate_runtime": sum(row["candidate_runtime"] for row in members),
        "runtime_delta": sum(row["runtime_delta"] for row in members),
    }


def main() -> None:
    args = common_parser(__doc__, "03_preprocessing_impact").parse_args()
    data_root, manifest_path, output = resolve_args(args)
    manifest = load_manifest(manifest_path)
    expected = expected_page_sizes(manifest)
    stages, load_counts = load_stages(data_root, manifest)

    stage_summaries = []
    for model in MODELS:
        for stage in STAGES:
            words = word_metric_rows(
                stages[(model, stage)], manifest, model, stage
            )
            pages = page_metric_rows(words, expected)
            stage_summaries.append({
                "configuration": f"{model}/{stage}",
                "model": model,
                "stage": stage,
                **summarize_metrics(words, pages, len(manifest)),
            })

    all_word_pairs = []
    all_page_pairs = []
    exclusions = []
    summary_rows = []
    example_rows = []

    for model in MODELS:
        for comparison, baseline_stage, candidate_stage in COMPARISONS:
            baseline = stages[(model, baseline_stage)]
            candidate = stages[(model, candidate_stage)]
            common_keys = sorted(set(baseline) & set(candidate))

            for key in sorted(set(baseline) - set(candidate)):
                exclusions.append({
                    "model": model,
                    "comparison": comparison,
                    "image_key": "/".join(key),
                    "excluded_stage": candidate_stage,
                    "reason": f"missing {candidate_stage} record",
                })
            for key in sorted(set(candidate) - set(baseline)):
                exclusions.append({
                    "model": model,
                    "comparison": comparison,
                    "image_key": "/".join(key),
                    "excluded_stage": baseline_stage,
                    "reason": f"missing {baseline_stage} record",
                })

            comparison_words = []
            grouped = defaultdict(list)
            for key in common_keys:
                row = build_word_pair(
                    model,
                    comparison,
                    baseline_stage,
                    candidate_stage,
                    key,
                    manifest[key]["ground_truth_text"],
                    manifest[key]["page_key"],
                    baseline[key],
                    candidate[key],
                )
                comparison_words.append(row)
                all_word_pairs.append(row)
                grouped[row["page_key"]].append(row)

            comparison_pages = [
                build_page_pair(model, comparison, members)
                for members in grouped.values()
            ]
            all_page_pairs.extend(comparison_pages)

            for level, rows in (
                ("word", comparison_words),
                ("page", comparison_pages),
            ):
                for metric, lower_is_better in (
                    ("cer", True),
                    ("wer", True),
                    ("exact_match", False),
                    ("missing_output", True),
                    ("runtime", True),
                ):
                    summary_rows.append(summarize_delta(
                        model,
                        comparison,
                        baseline_stage,
                        candidate_stage,
                        level,
                        metric,
                        rows,
                        args.seed,
                        lower_is_better,
                    ))

            ordered = sorted(comparison_words, key=lambda row: row["cer_delta"])
            for label, selected in (
                ("largest_improvement", ordered[:10]),
                ("largest_regression", list(reversed(ordered[-10:]))),
            ):
                for row in selected:
                    example_rows.append({"example_type": label, **row})

        # LLM validation exists for raw and final processed run_2 only. It is
        # intentionally not imputed for run_1.
        score_root = data_root / "llm_validations"
        raw_scores = {
            row["page_key"]: float(row["score"])
            for row in load_csv(score_root / model / "raw/scores.csv")
        }
        run_2_scores = {
            row["page_key"]: float(row["score"])
            for row in load_csv(score_root / model / "processed/scores.csv")
        }
        llm_rows = [{
            "baseline_llm_score": raw_scores[key],
            "candidate_llm_score": run_2_scores[key],
            "llm_score_delta": run_2_scores[key] - raw_scores[key],
        } for key in raw_scores.keys() & run_2_scores.keys()]
        if llm_rows:
            summary_rows.append(summarize_delta(
                model, "raw_to_run_2", "raw", "run_2", "page",
                "llm_score", llm_rows, args.seed, False,
            ))

        error_root = score_root / "error_detection" / model
        raw_errors = {
            row["page_key"]: float(row["flagged_region_rate"])
            for row in load_csv(error_root / "raw/page_results.csv")
        }
        run_2_errors = {
            row["page_key"]: float(row["flagged_region_rate"])
            for row in load_csv(error_root / "processed/page_results.csv")
        }
        error_rows = [{
            "baseline_flagged_region_rate": raw_errors[key],
            "candidate_flagged_region_rate": run_2_errors[key],
            "flagged_region_rate_delta": run_2_errors[key] - raw_errors[key],
        } for key in raw_errors.keys() & run_2_errors.keys()]
        if error_rows:
            summary_rows.append(summarize_delta(
                model, "raw_to_run_2", "raw", "run_2", "page",
                "flagged_region_rate", error_rows, args.seed, True,
            ))

    write_csv(output / "stage_overall_summary.csv", stage_summaries)
    write_csv(output / "paired_word_deltas.csv", all_word_pairs)
    write_csv(output / "paired_page_deltas.csv", all_page_pairs)
    write_csv(
        output / "excluded_unpaired_records.csv",
        exclusions,
        ["model", "comparison", "image_key", "excluded_stage", "reason"],
    )
    write_csv(output / "impact_summary.csv", summary_rows)
    write_csv(output / "representative_examples.csv", example_rows)

    svg_bar(
        output / "all_stage_accuracy.svg",
        "OCR accuracy and coverage across raw, run 1, and run 2",
        stage_summaries,
        "configuration",
        [
            ("corpus_cer", "CER", "#c53030"),
            ("corpus_wer", "WER", "#805ad5"),
            ("coverage", "Coverage", "#2f855a"),
        ],
    )
    page_accuracy = [
        row for row in summary_rows
        if row["level"] == "page" and row["metric"] in {"cer", "wer"}
    ]
    chart_rows = []
    for model in MODELS:
        for comparison, baseline_stage, candidate_stage in COMPARISONS:
            row = {"comparison": f"{model}/{comparison}"}
            for metric in ("cer", "wer"):
                match = next(
                    item for item in page_accuracy
                    if item["model"] == model
                    and item["comparison"] == comparison
                    and item["metric"] == metric
                )
                row[f"baseline_{metric}"] = match["baseline_mean"]
                row[f"candidate_{metric}"] = match["candidate_mean"]
            chart_rows.append(row)
    svg_bar(
        output / "paired_page_accuracy.svg",
        "Paired improvement for each preprocessing transition",
        chart_rows,
        "comparison",
        [
            ("baseline_cer", "Baseline CER", "#e53e3e"),
            ("candidate_cer", "Candidate CER", "#38a169"),
            ("baseline_wer", "Baseline WER", "#805ad5"),
            ("candidate_wer", "Candidate WER", "#3182ce"),
        ],
    )
    scatter_colors = {
        f"{model}/{comparison}": color
        for model, color in (("surya", "#2b6cb0"), ("tesseract", "#dd6b20"))
        for comparison, _, _ in COMPARISONS
    }
    for row in all_word_pairs:
        row["comparison_series"] = f"{row['model']}/{row['comparison']}"
    svg_scatter(
        output / "baseline_vs_candidate_cer.svg",
        "Paired word CER for all preprocessing transitions",
        all_word_pairs,
        "baseline_cer",
        "candidate_cer",
        "comparison_series",
        scatter_colors,
        "Baseline CER",
        "Candidate CER",
    )

    transition_counts = {
        model: {
            comparison: {
                "paired_words": sum(
                    row["model"] == model and row["comparison"] == comparison
                    for row in all_word_pairs
                ),
                "paired_pages": sum(
                    row["model"] == model and row["comparison"] == comparison
                    for row in all_page_pairs
                ),
                "excluded_records": sum(
                    row["model"] == model and row["comparison"] == comparison
                    for row in exclusions
                ),
            }
            for comparison, _, _ in COMPARISONS
        }
        for model in MODELS
    }
    write_json(output / "summary.json", {
        "stages": list(STAGES),
        "comparisons": [comparison for comparison, _, _ in COMPARISONS],
        "pairing_rule": "same split/writer/page/image identity within each model and transition",
        "transition_counts": transition_counts,
        "load_counts": {
            f"{model}/{stage}": value
            for (model, stage), value in load_counts.items()
        },
        "llm_validation_scope": (
            "LLM scores and detected-error rates are available only for "
            "raw versus final run_2; no run_1 values are imputed."
        ),
        "bootstrap_iterations": 5000,
        "seed": args.seed,
        "paired_test": "two-sided Wilcoxon signed-rank",
    })
    report_rows = [
        row for row in summary_rows
        if row["level"] == "page" and row["metric"] in {"cer", "wer"}
    ]
    (output / "report_section.md").write_text(
        "# Preprocessing impact\n\n"
        + markdown_table(report_rows, [
            "model", "comparison", "metric", "pairs", "baseline_mean",
            "candidate_mean", "absolute_change", "bootstrap_ci_low",
            "bootstrap_ci_high", "wins", "ties", "losses", "p_value",
        ])
        + "\nThe analysis reports raw→run 1, run 1→run 2, and raw→run 2 "
          "on separate identity-paired cohorts. Negative CER/WER changes indicate "
          "improvement. Overall stage summaries are reported separately because "
          "Surya coverage differs between stages. LLM validation was not run for "
          "run 1 and is therefore shown only for raw→run 2.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
