#!/usr/bin/env python3
"""Shared data loading, metrics, statistics, and SVG helpers for Phase 5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
import unicodedata
from typing import Iterable, Sequence

import numpy as np
from scipy import stats
from jiwer import cer as jiwer_cer, wer as jiwer_wer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/ground_truth/manifests/ground_truth_manifest.csv"
COMPLETED = {"ok", "empty_output"}
CONFIGS = {
    ("surya", "raw"): "surya/raw/results.jsonl",
    ("surya", "processed"): "surya/processed/run_2/results.jsonl",
    ("tesseract", "raw"): "tesseract/raw/results.jsonl",
    ("tesseract", "processed"): "tesseract/processed/run_2/results.jsonl",
}
COLORS = {"surya": "#2b6cb0", "tesseract": "#dd6b20", "raw": "#718096", "processed": "#38a169"}


def normalize(value: object) -> str:
    return unicodedata.normalize("NFC", " ".join(str(value or "").split()))


def image_key(value: object) -> tuple[str, str, str, str] | None:
    parts = Path(str(value or "")).with_suffix("").parts
    if len(parts) == 4:
        return tuple(parts)
    for marker in ("sample_set", "ground_truth_processed_0622", "processed_sample_set"):
        if marker in parts:
            relative = parts[parts.index(marker) + 1 :]
            if len(relative) == 4:
                return tuple(relative)
    if len(parts) >= 4 and parts[-4] in {"train", "val", "test"}:
        return tuple(parts[-4:])
    return None


def key_string(key: tuple[str, str, str, str]) -> str:
    return "/".join(key)


def page_key(key: tuple[str, str, str, str]) -> str:
    return "_".join(key[:3])


def natural_key(key: tuple[str, str, str, str]) -> tuple:
    split, writer, page, image = key
    image_number = image.split("_", 1)[0]
    return split, int(writer), int(page), int(image_number)


def load_manifest(path: Path) -> dict[tuple[str, str, str, str], dict]:
    required = {"page_key", "split", "writer_id", "page_id", "image_index", "image_path", "ground_truth_text"}
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            key = image_key(row["image_path"])
            if key is None:
                raise ValueError(f"cannot derive image identity: {row['image_path']}")
            row["ground_truth_text"] = normalize(row["ground_truth_text"])
            rows[key] = row
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def load_results(path: Path, manifest: dict | None = None) -> tuple[dict, dict]:
    latest: dict[tuple[str, str, str, str], dict] = {}
    counts = {"lines": 0, "malformed": 0, "unmapped": 0, "noncompleted": 0}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            counts["lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed"] += 1
                continue
            key = image_key(record.get("relative_input_path")) or image_key(record.get("input_path"))
            if key is None or (manifest is not None and key not in manifest):
                counts["unmapped"] += 1
                continue
            if record.get("status") not in COMPLETED:
                counts["noncompleted"] += 1
                continue
            record = dict(record)
            record["text"] = normalize(record.get("text"))
            latest[key] = record
    return latest, counts


def config_path(data_root: Path, model: str, input_kind: str) -> Path:
    return data_root / "processed_ocr_responses" / CONFIGS[(model, input_kind)]


def load_configs(data_root: Path, manifest: dict) -> tuple[dict, dict]:
    configs, load_counts = {}, {}
    for key in CONFIGS:
        path = config_path(data_root, *key)
        if not path.is_file():
            raise FileNotFoundError(f"required OCR results missing: {path}")
        configs[key], load_counts[key] = load_results(path, manifest)
    return configs, load_counts


def edit_alignment(reference: Sequence, hypothesis: Sequence) -> tuple[int, list[tuple[str, object, object]]]:
    """Return Levenshtein distance and a deterministic forward edit alignment."""
    n, m = len(reference), len(hypothesis)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    operations = []
    i, j = n, m
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and table[i][j] == table[i - 1][j - 1]:
            operations.append(("equal", reference[i - 1], hypothesis[j - 1])); i -= 1; j -= 1
        elif i and j and table[i][j] == table[i - 1][j - 1] + 1:
            operations.append(("substitution", reference[i - 1], hypothesis[j - 1])); i -= 1; j -= 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            operations.append(("deletion", reference[i - 1], "")); i -= 1
        else:
            operations.append(("insertion", "", hypothesis[j - 1])); j -= 1
    operations.reverse()
    return table[n][m], operations


def text_metrics(reference: str, hypothesis: str) -> dict[str, float | bool]:
    return {
        "cer": float(jiwer_cer(reference, hypothesis)),
        "wer": float(jiwer_wer(reference, hypothesis)),
        "exact_match": reference == hypothesis,
    }


def word_metric_rows(records: dict, manifest: dict, model: str, input_kind: str) -> list[dict]:
    rows = []
    for key, record in sorted(records.items(), key=lambda item: natural_key(item[0])):
        reference = manifest[key]["ground_truth_text"]
        hypothesis = record["text"]
        metric = text_metrics(reference, hypothesis)
        rows.append({
            "image_key": key_string(key), "page_key": manifest[key]["page_key"],
            "split": key[0], "writer_id": key[1], "page_id": key[2], "image_index": key[3],
            "model": model, "input_kind": input_kind, "reference": reference, "hypothesis": hypothesis,
            "cer": metric["cer"], "wer": metric["wer"], "exact_match": metric["exact_match"],
            "missing_output": not hypothesis, "status": record.get("status", ""),
            "elapsed_seconds": float(record.get("elapsed_seconds") or 0),
            "source_image_path": manifest[key]["image_path"], "input_image_path": record.get("input_path", ""),
        })
    return rows


def page_metric_rows(word_rows: list[dict], expected: dict[str, int]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in word_rows:
        groups.setdefault(row["page_key"], []).append(row)
    pages = []
    for pkey, members in sorted(groups.items()):
        members.sort(key=lambda row: int(str(row["image_index"]).split("_", 1)[0]))
        reference = normalize(" ".join(row["reference"] for row in members))
        hypothesis = normalize(" ".join(row["hypothesis"] for row in members))
        metric = text_metrics(reference, hypothesis)
        first = members[0]
        pages.append({
            "page_key": pkey, "split": first["split"], "writer_id": first["writer_id"], "page_id": first["page_id"],
            "model": first["model"], "input_kind": first["input_kind"], "reference": reference, "hypothesis": hypothesis,
            "cer": metric["cer"], "wer": metric["wer"], "exact_match": metric["exact_match"],
            "completed_words": len(members), "expected_words": expected[pkey],
            "coverage": len(members) / expected[pkey],
            "missing_output_rate": sum(row["missing_output"] for row in members) / len(members),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in members),
        })
    return pages


def expected_page_sizes(manifest: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in manifest.values():
        counts[row["page_key"]] = counts.get(row["page_key"], 0) + 1
    return counts


def summarize_metrics(word_rows: list[dict], page_rows: list[dict], manifest_size: int) -> dict:
    refs = [row["reference"] for row in word_rows]
    hyps = [row["hypothesis"] for row in word_rows]
    corpus_ref, corpus_hyp = "\n".join(refs), "\n".join(hyps)
    corpus = text_metrics(corpus_ref, corpus_hyp)
    return {
        "records": len(word_rows), "manifest_records": manifest_size, "coverage": len(word_rows) / manifest_size,
        "pages": len(page_rows), "complete_pages": sum(row["coverage"] == 1 for row in page_rows),
        "corpus_cer": corpus["cer"], "corpus_wer": corpus["wer"],
        "mean_word_cer": mean(row["cer"] for row in word_rows), "mean_word_wer": mean(row["wer"] for row in word_rows),
        "mean_page_cer": mean(row["cer"] for row in page_rows), "mean_page_wer": mean(row["wer"] for row in page_rows),
        "exact_accuracy": mean(float(row["exact_match"]) for row in word_rows),
        "missing_output_rate": mean(float(row["missing_output"]) for row in word_rows),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else float("nan")


def load_csv(path: Path, required: Iterable[str] = ()) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as target:
        if not fields:
            target.write("")
            return
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_parser(description: str, default_subdir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(default_subdir=default_subdir)
    return parser


def resolve_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_root = args.data_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = (args.output_dir or data_root / "phase_5" / args.default_subdir).expanduser().resolve()
    try:
        output.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"--output-dir must be beneath --data-root ({data_root})") from exc
    output.mkdir(parents=True, exist_ok=True)
    return data_root, manifest, output


def bootstrap_mean_ci(values: Sequence[float], seed: int, iterations: int = 5000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    # Chunking bounds memory while retaining a full 5,000-draw bootstrap.
    chunk = 500
    for start in range(0, iterations, chunk):
        count = min(chunk, iterations-start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        samples[start:start+count] = array[indices].mean(axis=1)
    return percentile(samples, 2.5), percentile(samples, 97.5)


def paired_test(values: Sequence[float]) -> dict:
    if not values or all(abs(value) < 1e-15 for value in values):
        return {"test": "wilcoxon", "statistic": 0.0, "p_value": 1.0}
    result = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided")
    return {"test": "wilcoxon", "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def svg_frame(title: str, body: str, width: int = 900, height: int = 520) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">{html.escape(title)}</text>'
            f'{body}</svg>\n')


def svg_bar(path: Path, title: str, rows: list[dict], label: str, series: list[tuple[str, str, str]], y_max: float | None = None) -> None:
    width, height, left, top, bottom = 900, 520, 90, 55, 100
    plot_w, plot_h = width-left-30, height-top-bottom
    maximum = y_max or max([float(row[key]) for row in rows for key, _, _ in series] + [1])
    group_w = plot_w / max(1, len(rows)); bar_w = group_w / (len(series)+1)
    body = [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>', f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>']
    for tick in range(6):
        value = maximum*tick/5; y = top+plot_h-(value/maximum*plot_h)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>')
    for i, row in enumerate(rows):
        center = left+(i+.5)*group_w
        body.append(f'<text x="{center:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(str(row[label]))}</text>')
        for j, (key, name, color) in enumerate(series):
            value = float(row[key]); h = value/maximum*plot_h; x = center-(len(series)*bar_w)/2+j*bar_w
            body.append(f'<rect x="{x:.1f}" y="{top+plot_h-h:.1f}" width="{bar_w*.82:.1f}" height="{h:.1f}" fill="{color}"/><text x="{x+bar_w*.41:.1f}" y="{top+plot_h-h-4:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.3f}</text>')
    for j, (_, name, color) in enumerate(series):
        x = left+j*155; body.append(f'<rect x="{x}" y="{height-32}" width="14" height="14" fill="{color}"/><text x="{x+20}" y="{height-20}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
    path.write_text(svg_frame(title, "".join(body), width, height), encoding="utf-8")


def svg_scatter(path: Path, title: str, points: list[dict], x_key: str, y_key: str, label_key: str, colors: dict[str, str], x_label: str, y_label: str) -> None:
    width, height, left, top, bottom, right = 900, 520, 90, 55, 70, 35
    xs = [float(p[x_key]) for p in points]; ys = [float(p[y_key]) for p in points]
    xmin, xmax = (min(xs), max(xs)) if xs else (0, 1); ymin, ymax = (min(ys), max(ys)) if ys else (0, 1)
    if xmax == xmin: xmax += 1
    if ymax == ymin: ymax += 1
    pw, ph = width-left-right, height-top-bottom
    body = [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/><line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>']
    for point in points:
        x = left+(float(point[x_key])-xmin)/(xmax-xmin)*pw; y = top+ph-(float(point[y_key])-ymin)/(ymax-ymin)*ph
        label = str(point[label_key]); body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors.get(label, "#4a5568")}" fill-opacity="0.72"><title>{html.escape(label)}</title></circle>')
    body.append(f'<text x="{left+pw/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text><text x="20" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 20 {top+ph/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>')
    path.write_text(svg_frame(title, "".join(body), width, height), encoding="utf-8")


def svg_boxplot(path: Path, title: str, groups: list[tuple[str, Sequence[float], str]], y_min: float = 0, y_max: float = 5) -> None:
    width, height, left, top, bottom = 900, 520, 90, 55, 85
    pw, ph = width-left-30, height-top-bottom
    body = [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/><line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>']
    def ypos(value): return top+ph-(value-y_min)/(y_max-y_min)*ph
    for tick in range(int(y_min), int(y_max)+1):
        y = ypos(tick); body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e2e8f0"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}</text>')
    group_w = pw/max(1, len(groups))
    for i, (label, values, color) in enumerate(groups):
        values = list(values); x = left+(i+.5)*group_w
        if values:
            q1, median, q3 = np.percentile(values, [25, 50, 75]); low, high = min(values), max(values)
            body.extend([f'<line x1="{x:.1f}" y1="{ypos(low):.1f}" x2="{x:.1f}" y2="{ypos(high):.1f}" stroke="#333"/>', f'<rect x="{x-28:.1f}" y="{ypos(q3):.1f}" width="56" height="{ypos(q1)-ypos(q3):.1f}" fill="{color}" fill-opacity="0.55" stroke="#333"/>', f'<line x1="{x-28:.1f}" y1="{ypos(median):.1f}" x2="{x+28:.1f}" y2="{ypos(median):.1f}" stroke="#111" stroke-width="2"/>'])
        body.append(f'<text x="{x:.1f}" y="{top+ph+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(label)}</text>')
    path.write_text(svg_frame(title, "".join(body), width, height), encoding="utf-8")


def svg_heatmap(path: Path, title: str, rows: list[str], columns: list[str], values: dict[tuple[str, str], float]) -> None:
    width, height, left, top = 950, max(400, 130+45*len(rows)), 210, 80
    cell_w = (width-left-30)/max(1, len(columns)); cell_h = 42
    maximum = max(values.values(), default=1) or 1
    body = []
    for j, column in enumerate(columns):
        x = left+(j+.5)*cell_w; body.append(f'<text x="{x:.1f}" y="{top-12}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(column)}</text>')
    for i, row in enumerate(rows):
        y = top+i*cell_h; body.append(f'<text x="{left-10}" y="{y+26}" text-anchor="end" font-family="sans-serif" font-size="11">{html.escape(row)}</text>')
        for j, column in enumerate(columns):
            value = values.get((row, column), 0); intensity = value/maximum
            shade = int(245-175*intensity); color = f'rgb({shade},{shade+10},{255-int(shade*.25)})'
            x = left+j*cell_w; body.append(f'<rect x="{x:.1f}" y="{y}" width="{cell_w-2:.1f}" height="{cell_h-2}" fill="{color}"/><text x="{x+cell_w/2:.1f}" y="{y+25}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.3f}</text>')
    path.write_text(svg_frame(title, "".join(body), width, height), encoding="utf-8")


def markdown_table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return "_No rows available._\n"
    def value(row, field):
        item = row.get(field, "")
        return f"{item:.4f}" if isinstance(item, float) else str(item).replace("|", "\\|")
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    lines.extend("| " + " | ".join(value(row, field) for field in fields) + " |" for row in rows)
    return "\n".join(lines) + "\n"
