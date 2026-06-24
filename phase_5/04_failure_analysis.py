#!/usr/bin/env python3
"""Identify shared OCR failures and associate them with text/image properties."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import cv2
import numpy as np

from common import (
    common_parser, expected_page_sizes, load_configs, load_csv, load_manifest,
    markdown_table, mean, page_metric_rows, resolve_args, svg_bar, word_metric_rows,
    write_csv, write_json,
)
from importlib import import_module


VIRAMA = "్"
VOWEL_SIGNS = set(chr(code) for code in range(0x0C3E, 0x0C57)) | {"ం", "ః", "ఁ"}


def image_features(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    height, width = image.shape
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return {"width": width, "height": height, "aspect_ratio": width/max(1, height), "mean_intensity": float(image.mean()), "contrast_std": float(image.std()), "blur_laplacian_variance": float(laplacian.var()), "foreground_density": float(np.mean(threshold == 0))}


def standardized_difference(failures: list[float], others: list[float]) -> float:
    if not failures or not others:
        return float("nan")
    variance = ((len(failures)-1)*np.var(failures, ddof=1) + (len(others)-1)*np.var(others, ddof=1)) / max(1, len(failures)+len(others)-2)
    return (mean(failures)-mean(others))/math.sqrt(variance) if variance > 0 else 0.0


def main() -> None:
    args = common_parser(__doc__, "04_failure_analysis").parse_args()
    data_root, manifest_path, output = resolve_args(args)
    manifest = load_manifest(manifest_path); expected = expected_page_sizes(manifest)
    configs, _ = load_configs(data_root, manifest)
    processed_pages = {}
    for model in ("surya", "tesseract"):
        words = word_metric_rows(configs[(model, "processed")], manifest, model, "processed")
        for row in page_metric_rows(words, expected): processed_pages[(model, row["page_key"])] = row
    score_root = data_root / "llm_validations"
    llm_scores, detection_rates = {}, {}
    for model in ("surya", "tesseract"):
        for row in load_csv(score_root/model/"processed/scores.csv"): llm_scores[(model, row["page_key"])] = float(row["score"])
        for row in load_csv(score_root/"error_detection"/model/"processed/page_results.csv"): detection_rates[(model, row["page_key"])] = float(row["flagged_region_rate"])

    manifest_pages = defaultdict(list)
    for key, row in manifest.items(): manifest_pages[row["page_key"]].append((key, row))
    feature_rows = []
    for pkey, members in sorted(manifest_pages.items()):
        images = []
        for _, row in members:
            path = Path(row["image_path"])
            if not path.is_absolute(): path = Path(__file__).resolve().parents[2] / path
            images.append((path, image_features(path)))
        reference = " ".join(row["ground_truth_text"] for _, row in members)
        values = {name: mean(features[name] for _, features in images) for name in ("width", "height", "aspect_ratio", "mean_intensity", "contrast_std", "blur_laplacian_variance", "foreground_density")}
        values.update({"page_key": pkey, "split": members[0][1]["split"], "writer_id": members[0][1]["writer_id"], "page_id": members[0][1]["page_id"], "word_count": len(members), "reference_length": len(reference), "conjunct_density": reference.count(VIRAMA)/max(1, len(reference)), "diacritic_density": sum(char in VOWEL_SIGNS for char in reference)/max(1, len(reference)), "source_image_paths": ";".join(str(path) for path, _ in images)})
        for model in ("surya", "tesseract"):
            metric = processed_pages[(model, pkey)]
            values[f"{model}_cer"] = metric["cer"]
            values[f"{model}_wer"] = metric["wer"]
            values[f"{model}_missing_output_rate"] = metric["missing_output_rate"]
            values[f"{model}_llm_score"] = llm_scores.get((model, pkey), float("nan"))
            values[f"{model}_detected_error_rate"] = detection_rates.get((model, pkey), float("nan"))
            llm_bad = values[f"{model}_llm_score"] <= 2 if not math.isnan(values[f"{model}_llm_score"]) else False
            values[f"{model}_failure"] = metric["cer"] >= 0.8 and (llm_bad or metric["missing_output_rate"] >= 0.3)
        values["shared_failure"] = values["surya_failure"] and values["tesseract_failure"]
        reasons = []
        if values["shared_failure"]:
            if values["conjunct_density"] > mean(("్" in row["ground_truth_text"]) for _, row in members): reasons.append("conjunct-rich text")
            if values["blur_laplacian_variance"] < 100: reasons.append("low edge sharpness")
            if values["contrast_std"] < 45: reasons.append("low contrast")
            if max(values["surya_missing_output_rate"], values["tesseract_missing_output_rate"]) >= .3: reasons.append("many empty OCR regions")
            if not reasons: reasons.append("high error across both processed OCR models")
        values["likely_explanation"] = "; ".join(reasons)
        feature_rows.append(values)

    features = ["word_count", "reference_length", "conjunct_density", "diacritic_density", "width", "height", "aspect_ratio", "mean_intensity", "contrast_std", "blur_laplacian_variance", "foreground_density"]
    failure_rows = [row for row in feature_rows if row["shared_failure"]]; other_rows = [row for row in feature_rows if not row["shared_failure"]]
    associations = []
    for feature in features:
        fail_values = [float(row[feature]) for row in failure_rows]; other_values = [float(row[feature]) for row in other_rows]
        associations.append({"feature": feature, "failure_mean": mean(fail_values), "nonfailure_mean": mean(other_values), "standardized_mean_difference": standardized_difference(fail_values, other_values), "absolute_association_rank_value": abs(standardized_difference(fail_values, other_values))})
    associations.sort(key=lambda row: row["absolute_association_rank_value"], reverse=True)
    for rank, row in enumerate(associations, 1): row["association_rank"] = rank
    review = [{key: row[key] for key in ["page_key", "split", "writer_id", "page_id", "source_image_paths", "surya_cer", "tesseract_cer", "surya_llm_score", "tesseract_llm_score", "surya_missing_output_rate", "tesseract_missing_output_rate", "conjunct_density", "diacritic_density", "contrast_std", "blur_laplacian_variance", "foreground_density", "likely_explanation"]} for row in failure_rows]
    write_csv(output / "page_failure_features.csv", feature_rows)
    write_csv(output / "feature_associations.csv", associations)
    write_csv(output / "shared_failure_review.csv", review, list(review[0]) if review else ["page_key"])
    chart = [{"feature": row["feature"], "association": row["absolute_association_rank_value"]} for row in associations[:8]]
    svg_bar(output / "failure_feature_associations.svg", "Strongest descriptive associations with shared failure", chart, "feature", [("association", "Absolute standardized difference", "#c53030")])
    rule = "For each processed model: page CER >= 0.80 AND (GPT-4o score <= 2 OR missing-output rate >= 0.30). A shared failure satisfies the rule for both models."
    write_json(output / "summary.json", {"failure_rule": rule, "evaluated_pages": len(feature_rows), "shared_failure_pages": len(failure_rows), "nonfailure_pages": len(other_rows), "interpretation_limit": "Associations are descriptive and do not establish causation.", "future_directions": ["Improve word segmentation and crop margins", "Select preprocessing condition per image rather than globally", "Ensemble Surya and Tesseract with confidence-aware routing", "Route high-risk pages to human review", "Train or adapt a Telugu handwriting recognizer"]})
    (output / "report_section.md").write_text("# Failure analysis\n\n**Failure rule:** " + rule + f"\n\nShared failures: {len(failure_rows)} of {len(feature_rows)} pages.\n\n" + markdown_table(associations[:8], ["association_rank", "feature", "failure_mean", "nonfailure_mean", "standardized_mean_difference"]) + "\nThese associations describe the evaluated sample; they are not causal estimates. Telugu conjunct and vowel-sign densities are reported explicitly to connect failures to script structure.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
