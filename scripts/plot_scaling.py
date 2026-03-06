#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
"""
Read Google Benchmark JSON output from scaling sweeps and produce a Markdown
summary table.  Optionally generates matplotlib plots if matplotlib is
available.

Usage:
    python3 scripts/plot_scaling.py <results_dir>

Example:
    python3 scripts/plot_scaling.py build/scaling_results
"""

import json
import sys
from pathlib import Path

def load_results(results_dir: Path) -> dict:
    """Load all scaling_*ranks.json files, keyed by rank count."""
    data = {}
    for path in sorted(results_dir.glob("scaling_*ranks.json")):
        stem = path.stem  # e.g. "scaling_4ranks"
        ranks = int(stem.split("_")[1].replace("ranks", ""))
        with open(path) as f:
            data[ranks] = json.load(f)
    return data


def extract_benchmarks(all_data: dict) -> dict:
    """Extract per-benchmark, per-rank-count timing and counters."""
    benchmarks: dict = {}
    for ranks, result in sorted(all_data.items()):
        for bm in result.get("benchmarks", []):
            name = bm["name"]
            if name not in benchmarks:
                benchmarks[name] = []
            benchmarks[name].append({
                "ranks": ranks,
                "real_time": bm.get("real_time", 0),
                "cpu_time": bm.get("cpu_time", 0),
                "time_unit": bm.get("time_unit", "ns"),
                "bytes_per_second": bm.get("bytes_per_second", 0),
                **{k: v for k, v in bm.items() if k in (
                    "ranks", "msg_bytes", "global_elements",
                    "per_rank_elements", "efficiency")},
            })
    return benchmarks


def compute_efficiency(entries: list) -> list:
    """Add 'efficiency' field: T(1) / (N * T(N))."""
    if not entries:
        return entries
    baseline = None
    for e in entries:
        if e["ranks"] == 1:
            baseline = e["real_time"]
            break
    if baseline is None:
        # Use smallest rank count as baseline
        entries_sorted = sorted(entries, key=lambda x: x["ranks"])
        baseline = entries_sorted[0]["real_time"] * entries_sorted[0]["ranks"]
    for e in entries:
        if e["real_time"] > 0:
            e["efficiency"] = baseline / (e["ranks"] * e["real_time"])
        else:
            e["efficiency"] = 0.0
    return entries


def print_markdown(benchmarks: dict) -> None:
    """Print a Markdown table for each benchmark."""
    for name, entries in sorted(benchmarks.items()):
        entries = compute_efficiency(entries)
        print(f"\n### {name}\n")
        print("| Ranks | Time ({unit}) | Bytes/s | Efficiency |".format(
            unit=entries[0]["time_unit"] if entries else "us"))
        print("|------:|-------------:|--------:|-----------:|")
        for e in sorted(entries, key=lambda x: x["ranks"]):
            bps = e.get("bytes_per_second", 0)
            bps_str = f"{bps:.2e}" if bps else "N/A"
            print(f"| {e['ranks']:5d} | {e['real_time']:12.2f} | {bps_str:>7s} | {e['efficiency']:10.3f} |")


def try_plot(benchmarks: dict, results_dir: Path) -> None:
    """Generate matplotlib plots if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — skipping plot generation)")
        return

    for name, entries in sorted(benchmarks.items()):
        entries = compute_efficiency(sorted(entries, key=lambda x: x["ranks"]))
        ranks = [e["ranks"] for e in entries]
        times = [e["real_time"] for e in entries]
        effs = [e["efficiency"] for e in entries]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(name)

        ax1.plot(ranks, times, "o-")
        ax1.set_xlabel("Ranks")
        ax1.set_ylabel(f"Time ({entries[0]['time_unit']})")
        ax1.set_title("Execution Time")
        ax1.grid(True, alpha=0.3)

        ax2.plot(ranks, effs, "s-", color="green")
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Ranks")
        ax2.set_ylabel("Efficiency")
        ax2.set_title("Parallel Efficiency")
        ax2.set_ylim(0, max(1.2, max(effs) * 1.1))
        ax2.grid(True, alpha=0.3)

        safe_name = name.replace("/", "_").replace(" ", "_")
        out_path = results_dir / f"{safe_name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    all_data = load_results(results_dir)
    if not all_data:
        print(f"No scaling_*ranks.json files found in {results_dir}")
        sys.exit(1)

    print(f"# DTL Scaling Benchmark Results\n")
    print(f"Rank counts tested: {sorted(all_data.keys())}")

    benchmarks = extract_benchmarks(all_data)
    for entries in benchmarks.values():
        compute_efficiency(entries)

    print_markdown(benchmarks)
    try_plot(benchmarks, results_dir)


if __name__ == "__main__":
    main()
