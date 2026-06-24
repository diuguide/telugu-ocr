#!/usr/bin/env python3
"""Compare canonical Surya/Tesseract raw and processed OCR configurations."""

from __future__ import annotations

from pathlib import Path
import math

from common import (
    COLORS, common_parser, expected_page_sizes, load_configs, load_csv, load_manifest,
    markdown_table, mean, page_metric_rows, pearson, resolve_args, summarize_metrics,
    svg_bar, svg_boxplot, svg_scatter, word_metric_rows, write_csv, write_json,
)


def validation_paths(data_root: Path, model: str, kind: str) -> tuple[Path, Path]:
    root = data_root / "llm_validations"
    return root / model / kind / "scores.csv", root / "error_detection" / model / kind / "page_results.csv"


def main() -> None:
    args = common_parser(__doc__, "01_model_comparison").parse_args()
    data_root, manifest_path, output = resolve_args(args)
    manifest = load_manifest(manifest_path); expected = expected_page_sizes(manifest)
    configs, load_counts = load_configs(data_root, manifest)
    all_words, all_pages, summaries, score_groups = [], [], [], []
    page_lookup = {}
    correlations = []
    for (model, kind), records in configs.items():
        words = word_metric_rows(records, manifest, model, kind)
        pages = page_metric_rows(words, expected)
        all_words.extend(words); all_pages.extend(pages)
        page_lookup.update({(model, kind, row["page_key"]): row for row in pages})
        summary = {"configuration": f"{model}/{kind}", "model": model, "input_kind": kind, **summarize_metrics(words, pages, len(manifest))}
        score_path, errors_path = validation_paths(data_root, model, kind)
        scores = load_csv(score_path, ["page_key", "score"])
        errors = load_csv(errors_path, ["page_key", "flagged_region_rate"])
        summary["mean_llm_score"] = mean(float(row["score"]) for row in scores)
        summary["llm_pages"] = len(scores)
        summary["mean_flagged_region_rate"] = mean(float(row["flagged_region_rate"]) for row in errors)
        summary["error_detection_pages"] = len(errors)
        agreement_path = data_root / "llm_validations/cross_model_agreement" / kind / "summary.json"
        import json
        agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
        summary["mean_cross_model_page_agreement"] = float(agreement["mean_page_agreement"])
        summary["cross_model_pages"] = int(agreement["compared_pages"])
        summaries.append(summary)
        score_groups.append((f"{model}/{kind}", [float(row["score"]) for row in scores], COLORS[model]))
        joined = [(float(row["score"]), page_lookup[(model, kind, row["page_key"])]["cer"]) for row in scores if (model, kind, row["page_key"]) in page_lookup]
        coefficient, p_value = pearson([item[0] for item in joined], [item[1] for item in joined])
        correlations.append({"configuration": f"{model}/{kind}", "paired_pages": len(joined), "pearson_r": coefficient, "p_value": p_value})

    common_keys = set.intersection(*(set(records) for records in configs.values()))
    paired_rows = []
    for (model, kind), records in configs.items():
        words = word_metric_rows({key: records[key] for key in common_keys}, manifest, model, kind)
        pages = page_metric_rows(words, expected)
        paired_rows.append({"configuration": f"{model}/{kind}", "paired_records": len(common_keys), **summarize_metrics(words, pages, len(common_keys))})

    for row in summaries:
        row["accuracy_rank"] = 0
    for rank, row in enumerate(sorted(summaries, key=lambda item: (item["corpus_cer"], item["corpus_wer"], -item["coverage"])), 1):
        row["accuracy_rank"] = rank
    ranked = sorted(summaries, key=lambda item: item["accuracy_rank"])
    write_csv(output / "configuration_summary.csv", summaries)
    write_csv(output / "ranked_summary.csv", ranked)
    write_csv(output / "word_metrics.csv", all_words)
    write_csv(output / "page_metrics.csv", all_pages)
    write_csv(output / "paired_configuration_summary.csv", paired_rows)
    write_csv(output / "llm_cer_correlations.csv", correlations)
    svg_bar(output / "cer_wer_coverage.svg", "OCR accuracy and coverage", summaries, "configuration", [("corpus_cer", "CER", "#c53030"), ("corpus_wer", "WER", "#805ad5"), ("coverage", "Coverage", "#2f855a")])
    svg_boxplot(output / "llm_score_distribution.svg", "GPT-4o lexical-quality scores", score_groups, 0, 5)
    points = []
    score_root = data_root / "llm_validations"
    for model, kind in configs:
        for row in load_csv(score_root/model/kind/"scores.csv"):
            metric = page_lookup.get((model, kind, row["page_key"]))
            if metric: points.append({"score": float(row["score"]), "cer": metric["cer"], "configuration": f"{model}/{kind}"})
    svg_scatter(output / "cer_vs_llm_score.svg", "Page CER versus GPT-4o score", points, "score", "cer", "configuration", {f"{m}/{k}": COLORS[m] if k == "processed" else COLORS[k] for m, k in configs}, "GPT-4o score", "Page CER")
    write_json(output / "summary.json", {"canonical_configurations": [row["configuration"] for row in summaries], "paired_cohort_records": len(common_keys), "ranking": [row["configuration"] for row in ranked], "load_counts": {f"{m}/{k}": value for (m,k), value in load_counts.items()}, "correlations": correlations})
    note = "# Model comparison\n\n" + markdown_table(ranked, ["accuracy_rank", "configuration", "corpus_cer", "corpus_wer", "coverage", "mean_llm_score", "mean_flagged_region_rate"]) + f"\nThe strictly paired comparison contains {len(common_keys)} word regions. Overall and paired results are kept separate because Surya raw coverage is incomplete. CER, WER, and LLM lexical-quality scores measure different behavior and are not treated as interchangeable.\n"
    (output / "report_section.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
