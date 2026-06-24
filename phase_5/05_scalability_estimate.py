#!/usr/bin/env python3
"""Project OCR runtime, storage, validation tokens, and API cost to full corpus."""

from __future__ import annotations

import json
from pathlib import Path

from common import (
    common_parser, config_path, load_configs, load_manifest, mean, percentile,
    resolve_args, svg_bar, write_csv, write_json, markdown_table,
)


FULL_WORDS = 126_539
FULL_PAGES = 5_368
PRACTICAL_WORKERS = 4
UTILIZATION = 0.75


def estimate_api_cost(input_tokens: float, output_tokens: float, pricing: dict) -> float:
    return input_tokens/1_000_000*pricing["input_usd_per_million_tokens"] + output_tokens/1_000_000*pricing["output_usd_per_million_tokens"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: continue
    return rows


def main() -> None:
    parser = common_parser(__doc__, "05_scalability")
    parser.add_argument("--pricing", type=Path, default=Path(__file__).with_name("pricing_2026-06-23.json"))
    args = parser.parse_args()
    data_root, manifest_path, output = resolve_args(args)
    manifest = load_manifest(manifest_path); configs, _ = load_configs(data_root, manifest)
    pricing = json.loads(args.pricing.read_text(encoding="utf-8"))
    runtime_rows, projection_rows = [], []
    for (model, kind), records in configs.items():
        times = [float(row.get("elapsed_seconds") or 0) for row in records.values() if float(row.get("elapsed_seconds") or 0) > 0]
        result_path = config_path(data_root, model, kind)
        bytes_per_record = result_path.stat().st_size/max(1, len(records))
        row = {"configuration": f"{model}/{kind}", "observed_regions": len(times), "mean_seconds_per_region": mean(times), "median_seconds_per_region": percentile(times, 50), "p95_seconds_per_region": percentile(times, 95), "regions_per_hour_at_mean": 3600/mean(times), "observed_result_bytes_per_region": bytes_per_record}
        runtime_rows.append(row)
        for case, seconds in (("best_case_median", row["median_seconds_per_region"]), ("expected_mean", row["mean_seconds_per_region"]), ("conservative_p95", row["p95_seconds_per_region"])):
            serial = seconds*FULL_WORDS
            projection_rows.append({"configuration": row["configuration"], "scenario": case, "full_corpus_regions": FULL_WORDS, "full_corpus_pages": FULL_PAGES, "serial_seconds": serial, "serial_hours": serial/3600, "serial_days": serial/86400, "practical_workers": PRACTICAL_WORKERS, "utilization": UTILIZATION, "practical_hours": serial/(PRACTICAL_WORKERS*UTILIZATION*3600), "projected_result_gib": bytes_per_record*FULL_WORDS/(1024**3), "paid_ocr_api_cost_usd": 0.0})

    validation_rows, token_profiles = [], {}
    validation_roots = []
    for method, base in (("fluency", data_root/"llm_validations"), ("error_detection", data_root/"llm_validations/error_detection")):
        for model in ("surya", "tesseract"):
            for kind in ("raw", "processed"):
                path = base/model/kind/"results.jsonl"
                records = read_jsonl(path)
                usable = [row for row in records if row.get("usage")]
                elapsed = [float(row.get("elapsed_seconds") or 0) for row in usable]
                inputs = [int(row["usage"].get("input_tokens") or 0) for row in usable]
                outputs = [int(row["usage"].get("output_tokens") or 0) for row in usable]
                profile = {"method": method, "configuration": f"{model}/{kind}", "observed_pages": len(usable), "mean_seconds_per_page": mean(elapsed), "median_seconds_per_page": percentile(elapsed, 50), "p95_seconds_per_page": percentile(elapsed, 95), "mean_input_tokens_per_page": mean(inputs), "mean_output_tokens_per_page": mean(outputs)}
                validation_rows.append(profile); token_profiles[(method, model, kind)] = profile

    scenario_rows = []
    for name, pages in (("sampled_100_pages", 100), ("full_5368_pages", FULL_PAGES)):
        # Final reporting validates both processed OCR models with both LLM methods.
        selected = [token_profiles[(method, model, "processed")] for method in ("fluency", "error_detection") for model in ("surya", "tesseract")]
        input_tokens = sum(row["mean_input_tokens_per_page"]*pages for row in selected)
        output_tokens = sum(row["mean_output_tokens_per_page"]*pages for row in selected)
        elapsed = sum(row["mean_seconds_per_page"]*pages for row in selected)
        cost = estimate_api_cost(input_tokens, output_tokens, pricing)
        scenario_rows.append({"scenario": name, "pages_per_configuration": pages, "models": 2, "validation_methods": 2, "api_calls": pages*4, "estimated_input_tokens": round(input_tokens), "estimated_output_tokens": round(output_tokens), "estimated_standard_api_cost_usd": cost, "serial_validation_hours": elapsed/3600, "pricing_as_of": pricing["as_of_date"]})

    write_csv(output / "ocr_runtime_profile.csv", runtime_rows)
    write_csv(output / "ocr_full_corpus_projections.csv", projection_rows)
    write_csv(output / "validation_runtime_token_profile.csv", validation_rows)
    write_csv(output / "validation_cost_scenarios.csv", scenario_rows)
    write_json(output / "pricing_snapshot.json", pricing)
    write_json(output / "summary.json", {"full_corpus_inventory": {"word_images": FULL_WORDS, "page_directories": FULL_PAGES}, "runtime_basis": "elapsed_seconds from canonical OCR JSONL and Phase 4 validation JSONL", "practical_parallelism": {"workers": PRACTICAL_WORKERS, "utilization": UTILIZATION}, "local_ocr_cost_policy": "OCR monetary cost is reported as zero API cost; hardware, electricity, and labor are not monetized without measured rates.", "pricing": pricing, "scenarios": scenario_rows})
    svg_bar(output / "ocr_runtime_per_region.svg", "Observed OCR runtime per word region", runtime_rows, "configuration", [("median_seconds_per_region", "Median", "#38a169"), ("mean_seconds_per_region", "Mean", "#3182ce"), ("p95_seconds_per_region", "P95", "#c53030")])
    (output / "report_section.md").write_text("# Scalability estimate\n\nThe corpus inventory contains 126,539 word-region images in 5,368 page directories. OCR projections report serial time and a four-worker, 75%-utilization practical scenario. Local OCR compute is kept separate from paid GPT-4o validation.\n\n" + markdown_table(runtime_rows, ["configuration", "observed_regions", "median_seconds_per_region", "mean_seconds_per_region", "p95_seconds_per_region", "regions_per_hour_at_mean"]) + "\n" + markdown_table(scenario_rows, ["scenario", "api_calls", "estimated_input_tokens", "estimated_output_tokens", "estimated_standard_api_cost_usd", "serial_validation_hours", "pricing_as_of"]) + "\nPricing is a dated, reproducible snapshot and must be rechecked before a future production run.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
