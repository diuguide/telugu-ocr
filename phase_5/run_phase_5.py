#!/usr/bin/env python3
"""Run all Phase 5 analyses and assemble/render the report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from common import common_parser, config_path, file_sha256, resolve_args, write_json


ANALYSES = [
    ("01_model_comparison.py", "01_model_comparison"),
    ("02_error_analysis.py", "02_error_analysis"),
    ("03_preprocessing_impact.py", "03_preprocessing_impact"),
    ("04_failure_analysis.py", "04_failure_analysis"),
    ("05_scalability_estimate.py", "05_scalability"),
]


def build_report(output: Path) -> Path:
    sections = []
    for _, subdir in ANALYSES:
        sections.append((output/subdir/"report_section.md").read_text(encoding="utf-8"))
    figures = """
# Figures

![OCR accuracy and coverage](../01_model_comparison/cer_wer_coverage.svg){#fig-model-accuracy}

![GPT-4o score distributions](../01_model_comparison/llm_score_distribution.svg){#fig-llm-scores}

![CER and LLM-score relationship](../01_model_comparison/cer_vs_llm_score.svg){#fig-cer-llm}

![Deterministic error distribution](../02_error_analysis/error_distribution.svg){#fig-errors}

![Paired preprocessing comparison](../03_preprocessing_impact/paired_page_accuracy.svg){#fig-preprocessing}

![Failure-feature associations](../04_failure_analysis/failure_feature_associations.svg){#fig-failures}

![OCR runtime profile](../05_scalability/ocr_runtime_per_region.svg){#fig-runtime}
"""
    qmd = """---
title: "Phase 5 — Analysis, Insights, and Final Report"
author: "Telugu OCR Project"
date: last-modified
format:
  html:
    toc: true
    embed-resources: true
  pdf:
    pdf-engine: xelatex
    toc: true
    mainfont: "Noto Sans Telugu"
execute:
  enabled: false
---

# Scope and methodology

This report analyzes the canonical Surya and Tesseract OCR outputs on raw images and processed `run_2` images. The evaluation sample contains 961 labeled word regions across 40 dataset pages. Raw Surya coverage is incomplete, so overall summaries and strictly paired comparisons are reported separately. All generated evidence is traceable to machine-readable artifacts beneath `bin/data/phase_5`.

""" + "\n\n".join(sections) + figures + """
# Limitations and conclusions

The sample is deliberately stratified but small relative to the 5,368-page corpus. Surya raw coverage creates selection effects in unpaired summaries. GPT-4o scores and detections are validation signals rather than ground truth, while image-feature associations are descriptive rather than causal. Cost projections use observed latency and token use plus a dated public price snapshot; future production estimates should refresh those inputs.

The analysis therefore favors paired evidence, reports coverage beside accuracy, preserves regressions and exclusions, and provides review queues rather than overstating model reliability.
"""
    report_dir = output/"report"; report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir/"phase_5_report.qmd"; path.write_text(qmd, encoding="utf-8")
    return path


def main() -> None:
    # An empty subdirectory makes the runner's default exactly data/phase_5;
    # individual scripts still default to their numbered child directories.
    parser = common_parser(__doc__, "")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    data_root, manifest, output = resolve_args(args)
    script_dir = Path(__file__).resolve().parent
    commands = []
    for script, subdir in ANALYSES:
        command = [sys.executable, str(script_dir/script), "--data-root", str(data_root), "--manifest", str(manifest), "--output-dir", str(output/subdir), "--seed", str(args.seed)]
        subprocess.run(command, check=True); commands.append(command)
    report = build_report(output)
    rendered = []
    if not args.skip_render:
        env = os.environ.copy()
        cache = output/".quarto_cache"; cache.mkdir(parents=True, exist_ok=True)
        env.update({"XDG_CACHE_HOME": str(cache), "DENO_DIR": str(cache/"deno")})
        for target in ("html", "pdf"):
            subprocess.run(["quarto", "render", str(report), "--to", target], cwd=report.parent, env=env, check=True)
            rendered.append(str(report.with_suffix(f".{target}")))
    inputs = [manifest] + [config_path(data_root, model, kind) for model in ("surya", "tesseract") for kind in ("raw", "processed")]
    outputs = sorted(path for path in output.rglob("*") if path.is_file() and ".quarto_cache" not in path.parts)
    manifest_value = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(data_root), "ground_truth_manifest": str(manifest), "seed": args.seed, "commands": commands, "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path in inputs], "outputs": [{"path": str(path), "sha256": file_sha256(path)} for path in outputs], "rendered_reports": rendered}
    write_json(output/"analysis_manifest.json", manifest_value)
    print(f"Phase 5 complete: {output}")


if __name__ == "__main__":
    main()
