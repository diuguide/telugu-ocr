#!/usr/bin/env python3
"""Classify OCR errors with Unicode-aware alignments and GPT-4o evidence."""

from __future__ import annotations

from collections import Counter
import re
import unicodedata

from common import (
    COLORS, common_parser, edit_alignment, load_configs, load_csv, load_manifest,
    markdown_table, mean, normalize, resolve_args, svg_bar, svg_heatmap,
    text_metrics, write_csv, write_json,
)


VOWEL_SIGNS = set(chr(code) for code in range(0x0C3E, 0x0C57)) | {"ం", "ః", "ఁ"}
VIRAMA = "్"


def is_telugu(character: str) -> bool:
    return bool(character) and 0x0C00 <= ord(character) <= 0x0C7F


def classify_substitution(reference: str, hypothesis: str) -> str:
    if VIRAMA in {reference, hypothesis}:
        return "virama_or_conjunct_error"
    if reference in VOWEL_SIGNS or hypothesis in VOWEL_SIGNS:
        return "diacritic_or_vowel_sign_error"
    if reference.isdigit() or hypothesis.isdigit() or unicodedata.category(reference).startswith("P") or unicodedata.category(hypothesis).startswith("P"):
        return "punctuation_or_digit_error"
    return "base_character_substitution"


def classify_word(reference: str, hypothesis: str) -> tuple[set[str], list[dict]]:
    categories: set[str] = set(); events = []
    if not hypothesis:
        categories.add("missing_output")
    if reference.count(" ") != hypothesis.count(" "):
        categories.add("word_boundary_split_or_join")
    foreign = [char for char in hypothesis if char.isalpha() and not is_telugu(char)]
    if foreign:
        categories.add("non_telugu_output")
    _, operations = edit_alignment(list(reference), list(hypothesis))
    insertions = 0
    for operation, ref_char, hyp_char in operations:
        if operation == "equal":
            continue
        category = operation
        if operation == "substitution":
            category = classify_substitution(str(ref_char), str(hyp_char))
        elif operation == "deletion":
            if ref_char == VIRAMA: category = "virama_or_conjunct_error"
            elif ref_char in VOWEL_SIGNS: category = "diacritic_or_vowel_sign_error"
        elif operation == "insertion":
            insertions += 1
            if hyp_char == VIRAMA: category = "virama_or_conjunct_error"
            elif hyp_char in VOWEL_SIGNS: category = "diacritic_or_vowel_sign_error"
        categories.add(category)
        events.append({"operation": operation, "category": category, "reference_character": ref_char, "hypothesis_character": hyp_char})
    if insertions >= max(3, len(reference) // 2) or len(hypothesis) > max(4, int(len(reference) * 1.6)):
        categories.add("hallucinated_extra_text")
    return categories, events


def classify_llm_detection(row: dict) -> str:
    value = normalize(" ".join((row.get("error", ""), row.get("correction", ""), row.get("reason", "")))).lower()
    if "missing" in value or "omission" in value: return "missing_output"
    if "space" in value or "boundary" in value or "split" in value: return "word_boundary_split_or_join"
    if "hallucin" in value or "extra" in value: return "hallucinated_extra_text"
    if VIRAMA in value or "conjunct" in value or "virama" in value: return "virama_or_conjunct_error"
    if any(char in VOWEL_SIGNS for char in value) or "diacritic" in value or "vowel" in value: return "diacritic_or_vowel_sign_error"
    if re.search(r"[a-z]", value): return "non_telugu_output"
    return "llm_flagged_unspecified"


def main() -> None:
    args = common_parser(__doc__, "02_error_analysis").parse_args()
    data_root, manifest_path, output = resolve_args(args)
    manifest = load_manifest(manifest_path); configs, _ = load_configs(data_root, manifest)
    word_rows, event_rows, combined_evidence, subset_rows = [], [], [], []
    category_counts = Counter(); substitution_counts = Counter()
    config_totals = Counter()
    for (model, kind), records in configs.items():
        config = f"{model}/{kind}"
        conjunct_values, diacritic_values = [], []
        for key, record in records.items():
            reference, hypothesis = manifest[key]["ground_truth_text"], record["text"]
            categories, events = classify_word(reference, hypothesis)
            metric = text_metrics(reference, hypothesis); config_totals[config] += 1
            word = {"image_key": "/".join(key), "page_key": manifest[key]["page_key"], "model": model, "input_kind": kind, "reference": reference, "hypothesis": hypothesis, "cer": metric["cer"], "exact_match": metric["exact_match"], "categories": ";".join(sorted(categories)) or "correct", "error_category_count": len(categories)}
            word_rows.append(word)
            for category in categories:
                category_counts[(config, category)] += 1
                combined_evidence.append({**word, "evidence_source": "deterministic_alignment", "category": category, "error": "", "correction": "", "reason": "Unicode/edit alignment rule"})
            for event in events:
                event_row = {"image_key": "/".join(key), "page_key": manifest[key]["page_key"], "model": model, "input_kind": kind, **event}
                event_rows.append(event_row)
                if event["operation"] == "substitution": substitution_counts[(config, event["reference_character"], event["hypothesis_character"])] += 1
            if VIRAMA in reference: conjunct_values.append(metric)
            if any(char in VOWEL_SIGNS for char in reference): diacritic_values.append(metric)
        for subset, values in (("reference_contains_conjunct", conjunct_values), ("reference_contains_diacritic", diacritic_values)):
            subset_rows.append({"configuration": config, "subset": subset, "records": len(values), "mean_cer": mean(item["cer"] for item in values), "exact_accuracy": mean(float(item["exact_match"]) for item in values)})

        detection_path = data_root / "llm_validations/error_detection" / model / kind / "detections.csv"
        for row in load_csv(detection_path, ["page_key", "error", "correction", "reason"]):
            category = classify_llm_detection(row)
            combined_evidence.append({"image_key": "", "page_key": row["page_key"], "model": model, "input_kind": kind, "reference": "", "hypothesis": "", "cer": "", "exact_match": "", "categories": category, "error_category_count": 1, "evidence_source": "gpt4o_error_detection", "category": category, "error": row["error"], "correction": row["correction"], "reason": row["reason"]})

    summary_rows = []
    categories = sorted({category for _, category in category_counts})
    configurations = [f"{m}/{k}" for m, k in configs]
    for config in configurations:
        for category in categories:
            count = category_counts[(config, category)]
            summary_rows.append({"configuration": config, "category": category, "affected_words": count, "word_rate": count/config_totals[config]})
    substitution_rows = [{"configuration": config, "reference_character": ref, "hypothesis_character": hyp, "count": count} for (config, ref, hyp), count in substitution_counts.most_common()]
    write_csv(output / "word_error_categories.csv", word_rows)
    write_csv(output / "alignment_events.csv", event_rows)
    write_csv(output / "combined_error_evidence.csv", combined_evidence)
    write_csv(output / "error_category_summary.csv", summary_rows)
    write_csv(output / "frequent_substitutions.csv", substitution_rows)
    write_csv(output / "telugu_feature_subsets.csv", subset_rows)
    chart_rows = []
    for config in configurations:
        row = {"configuration": config}
        for category in categories: row[category] = category_counts[(config, category)]/config_totals[config]
        chart_rows.append(row)
    top_categories = sorted(categories, key=lambda cat: sum(category_counts[(cfg, cat)] for cfg in configurations), reverse=True)[:6]
    palette = ["#c53030", "#dd6b20", "#d69e2e", "#38a169", "#3182ce", "#805ad5"]
    svg_bar(output / "error_distribution.svg", "Most common deterministic OCR error categories", chart_rows, "configuration", [(category, category.replace("_", " "), palette[i]) for i, category in enumerate(top_categories)], y_max=1)
    svg_heatmap(output / "error_heatmap.svg", "Error rate by configuration", top_categories, configurations, {(category, config): category_counts[(config, category)]/config_totals[config] for category in top_categories for config in configurations})
    write_json(output / "summary.json", {"category_definitions": {"missing_output": "OCR returned no normalized Telugu text", "base_character_substitution": "Levenshtein substitution not involving a mark, virama, punctuation, or digit", "diacritic_or_vowel_sign_error": "Edit involving Telugu dependent vowel or combining sign", "virama_or_conjunct_error": "Edit involving U+0C4D", "insertion": "Inserted character", "deletion": "Deleted character", "word_boundary_split_or_join": "Reference and hypothesis space counts differ", "punctuation_or_digit_error": "Substitution involving punctuation or a digit", "non_telugu_output": "Hypothesis contains alphabetic characters outside Telugu", "hallucinated_extra_text": "Insertion-heavy or disproportionately long hypothesis"}, "configuration_records": dict(config_totals), "deterministic_evidence_rows": sum(1 for row in combined_evidence if row["evidence_source"] == "deterministic_alignment"), "gpt4o_evidence_rows": sum(1 for row in combined_evidence if row["evidence_source"] == "gpt4o_error_detection")})
    report = "# Error analysis\n\n" + markdown_table(summary_rows, ["configuration", "category", "affected_words", "word_rate"]) + "\nCategories are assigned from normalized Unicode alignments. GPT-4o detections are retained as a separate evidence source, avoiding false precision from treating LLM judgments as ground truth. Conjunct and diacritic subset results are reported separately.\n"
    (output / "report_section.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
