#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregate `request_benchmark_results.txt` produced by benchmark_script_fl.sh
# into a single CSV + Markdown table for reporting.
#
# Usage:
#   python3 benchmarks/ops/ascend/analysis.py --result-root /workspace/results
#   python3 benchmarks/ops/ascend/analysis.py --run-dirs /workspace/results/run_a /workspace/results/run_b
#   python3 benchmarks/ops/ascend/analysis.py --run-dirs ... --csv out.csv --md out.md

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List

RESULT_FILENAME = "request_benchmark_results.txt"
CASE_HEADER = re.compile(r"^=+ (i\d+_o\d+_np\d+_c\d+) =+$")

# "Mean TTFT (ms):   8441.34" -> ("Mean TTFT (ms)", "8441.34")
METRIC_LINE = re.compile(r"^([A-Za-z][^:]*?):\s+([-+]?[0-9.]*)\s*$")

FIELDS = [
    ("Successful requests", "successful"),
    ("Failed requests", "failed"),
    ("Maximum request concurrency", "concurrency"),
    ("Benchmark duration (s)", "duration_s"),
    ("Total input tokens", "input_tokens"),
    ("Total generated tokens", "output_tokens"),
    ("Request throughput (req/s)", "req_per_s"),
    ("Output token throughput (tok/s)", "output_tok_s"),
    ("Total token throughput (tok/s)", "total_tok_s"),
    ("Mean TTFT (ms)", "mean_ttft_ms"),
    ("Mean TPOT (ms)", "mean_tpot_ms"),
    ("Median TPOT (ms)", "median_tpot_ms"),
    ("Mean ITL (ms)", "mean_itl_ms"),
]

HEADER_FIELDS = ["run", "model", "model_path", "mode", "cudagraph_mode", "chunked_prefill", "case", "time"]


def _read_meta(lines: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    keys = {
        "Run": "run",
        "Model": "model",
        "Model path": "model_path",
        "Mode": "mode",
        "CUDAGRAPH_MODE": "cudagraph_mode",
        "Chunked prefill": "chunked_prefill",
    }
    for line in lines[:15]:
        for prefix, field in keys.items():
            if line.startswith(f"{prefix}:"):
                meta[field] = line.split(":", 1)[1].strip()
    return meta


def parse_file(path: str) -> List[Dict[str, str]]:
    """Parse one request_benchmark_results.txt into a row per (case, metric-set)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    meta = _read_meta(lines)
    rows: List[Dict[str, str]] = []
    current_case = ""
    current_time = ""
    metrics: Dict[str, str] = {}

    def flush() -> None:
        if not metrics:
            return
        row = dict(meta)
        row["case"] = current_case
        row["time"] = current_time
        for label, field in FIELDS:
            row[field] = metrics.get(label, "")
        rows.append(row)

    for line in lines:
        case_match = CASE_HEADER.match(line.strip())
        if case_match:
            flush()
            current_case = case_match.group(1)
            current_time = ""
            metrics = {}
            continue
        if line.startswith("TIME="):
            current_time = line.split("=", 1)[1].strip()
            continue
        if line.startswith("CASE_LOG="):
            continue
        metric_match = METRIC_LINE.match(line.strip())
        if metric_match and current_case:
            metrics[metric_match.group(1).strip()] = metric_match.group(2)

    flush()
    return rows


def discover_run_dirs(result_root: str) -> List[str]:
    if not os.path.isdir(result_root):
        raise SystemExit(f"result root not found: {result_root}")
    dirs = []
    for name in sorted(os.listdir(result_root)):
        candidate = os.path.join(result_root, name)
        if os.path.isfile(os.path.join(candidate, RESULT_FILENAME)):
            dirs.append(candidate)
    return dirs


def write_csv(rows: List[Dict[str, str]], path: str) -> None:
    columns = HEADER_FIELDS + [field for _, field in FIELDS]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: List[Dict[str, str]], path: str) -> None:
    header = ["run", "model", "mode", "case", "successful", "failed", "output_tok_s", "total_tok_s", "mean_ttft_ms", "mean_tpot_ms"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Ascend Op Benchmark Summary\n\n")
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("|" + "---|" * len(header) + "\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row.get(col, "")) for col in header) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Ascend op benchmark results")
    parser.add_argument("--result-root", default="/workspace/results", help="directory holding RUN_DIRs")
    parser.add_argument("--run-dirs", nargs="*", default=None, help="explicit RUN_DIR list (overrides --result-root)")
    parser.add_argument("--csv", default="benchmark_summary.csv", help="output CSV path")
    parser.add_argument("--md", default="benchmark_summary.md", help="output Markdown path")
    args = parser.parse_args()

    run_dirs = args.run_dirs or discover_run_dirs(args.result_root)
    rows: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        result_file = os.path.join(run_dir, RESULT_FILENAME)
        if not os.path.isfile(result_file):
            print(f"[SKIP] no {RESULT_FILENAME}: {run_dir}")
            continue
        parsed = parse_file(result_file)
        if not parsed:
            print(f"[WARN] no benchmark result parsed: {result_file}")
            continue
        rows.extend(parsed)
        print(f"[OK] {os.path.basename(run_dir)}: {len(parsed)} case(s)")

    if not rows:
        print("no data collected")
        return 1

    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    print(f"written: {args.csv}, {args.md} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
