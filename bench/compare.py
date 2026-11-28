#!/usr/bin/env python3
"""Compare two benchmark `report.json` runs and flag metric regressions.

Usage:
  python3 bench/compare.py --before path/to/old/report.json --after path/to/new/report.json
  python3 bench/compare.py --before old.json --after new.json --target rust --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _get(obj: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, bool) or cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


# (label, json_path_inside_suites, direction, tolerance_rel_or_abs, is_abs)
# direction: "lower_is_better" or "higher_is_better"
METRIC_SPECS: list[tuple[str, tuple[str, ...], str, float, bool]] = [
    ("Audio fan Voice-SNR gain (dB)", ("audio_quality", "{T}", "scenarios", "fan", "delta_voice_snr_db"), "higher_is_better", 1.0, True),
    ("Audio fan Delta-SNR (dB)", ("audio_quality", "{T}", "scenarios", "fan", "delta_snr_db"), "higher_is_better", 1.0, True),
    ("Audio fan Delta-SI-SDR (dB)", ("audio_quality", "{T}", "scenarios", "fan", "delta_si_sdr_db"), "higher_is_better", 1.0, True),
    ("Audio NR latency p99 (us)", ("audio_latency", "{T}", "workloads", "noise_removal", "latency_us", "p99"), "lower_is_better", 0.15, False),
    ("PCM16+RMS primitive p50 (us)", ("audio_latency", "{T}", "pcm_primitive", "latency_us", "p50"), "lower_is_better", 0.20, False),
    ("Idle daemon PSS (MB)", ("memory", "{T}", "checkpoints", "idle_standby", "pss_mb"), "lower_is_better", 0.10, False),
    ("Post-CUDA idle PSS (MB)", ("memory", "{T}", "checkpoints", "post_call_idle", "pss_mb"), "lower_is_better", 0.10, False),
    ("Video blur quality FPS", ("video", "{T}", "effects", "background_blur_quality", "fps"), "higher_is_better", 0.10, False),
    ("Config bench throughput (ops/s)", ("api_cli", "{T}", "config_bench", "ops_per_sec"), "higher_is_better", 0.15, False),
    ("CLI --version cold start p50 (ms)", ("api_cli", "{T}", "cli_cold_start_ms", "version", "p50"), "lower_is_better", 0.20, False),
    ("HTTP /api/state p95 (ms)", ("api_cli", "{T}", "http_endpoints", "GET /api/state", "latency_ms", "p95"), "lower_is_better", 0.15, False),
    ("Idle wakeups (ctxt/s)", ("system_impact", "{T}", "process_overhead", "total_ctxt_per_sec"), "lower_is_better", 0.20, False),
]


def compare_reports(
    before: dict[str, Any],
    after: dict[str, Any],
    targets: list[str] | None = None,
) -> dict[str, Any]:
    suites_b = before.get("suites", {})
    suites_a = after.get("suites", {})
    target_list = targets or list(dict.fromkeys([*before.get("targets", []), *after.get("targets", [])]))
    target_list = [t for t in target_list if t != "baseline"]

    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []

    for tgt in target_list:
        for label, raw_path, direction, tol, is_abs in METRIC_SPECS:
            path = tuple(tgt if part == "{T}" else part for part in raw_path)
            vb = _get(suites_b, path)
            va = _get(suites_a, path)
            if vb is None or va is None:
                continue
            delta = va - vb
            rel = (delta / abs(vb)) if abs(vb) > 1e-9 else 0.0

            regressed = False
            if direction == "higher_is_better":
                regressed = (delta < -tol) if is_abs else (rel < -tol)
            else:
                regressed = (delta > tol) if is_abs else (rel > tol)

            status = "REGRESSION" if regressed else "OK"
            row = {
                "target": tgt,
                "metric": label,
                "before": round(vb, 3),
                "after": round(va, 3),
                "delta": round(delta, 3),
                "pct_change": round(rel * 100.0, 2),
                "direction": direction,
                "status": status,
            }
            rows.append(row)
            if regressed:
                regressions.append(row)

        # Check reliability failure regression
        rb = (suites_b.get("reliability") or {}).get(tgt) or {}
        ra = (suites_a.get("reliability") or {}).get(tgt) or {}
        if rb.get("status") == "ok" and ra.get("status") == "ok":
            pb, pa = int(rb.get("passed", 0)), int(ra.get("passed", 0))
            ta = int(ra.get("total", 0))
            if pa < pb or pa < ta:
                reg = {
                    "target": tgt,
                    "metric": "Reliability checks passed",
                    "before": pb,
                    "after": pa,
                    "delta": pa - pb,
                    "pct_change": 0.0,
                    "direction": "higher_is_better",
                    "status": "REGRESSION",
                }
                rows.append(reg)
                regressions.append(reg)

    return {
        "before_meta": before.get("meta", {}),
        "after_meta": after.get("meta", {}),
        "rows": rows,
        "regressions": regressions,
        "ok": len(regressions) == 0,
    }


def format_diff_markdown(diff: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Regression Diff",
        "",
        "| Target | Metric | Before | After | Delta | Change | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in diff["rows"]:
        lines.append(
            f"| `{r['target']}` | {r['metric']} | {r['before']} | {r['after']} | "
            f"{r['delta']:+.3f} | {r['pct_change']:+.1f}% | **{r['status']}** |"
        )
    if not diff["rows"]:
        lines.append("| — | No overlapping metrics found | — | — | — | — | OK |")
    lines.append("")
    n_reg = len(diff["regressions"])
    summary = "PASS (0 regressions)" if diff["ok"] else f"FAIL ({n_reg} regressions)"
    lines.append(f"**Result:** {summary}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diff two Broadcast benchmark report.json files")
    p.add_argument("--before", required=True, help="Baseline report.json path")
    p.add_argument("--after", required=True, help="Candidate report.json path")
    p.add_argument("--target", default=None, help="Limit comparison to one target (e.g. python or rust)")
    p.add_argument("--json", action="store_true", help="Emit JSON output")
    args = p.parse_args(argv)

    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    targets = [args.target] if args.target else None
    diff = compare_reports(before, after, targets=targets)

    if args.json:
        print(json.dumps(diff, indent=2))
    else:
        print(format_diff_markdown(diff))
    return 0 if diff["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
