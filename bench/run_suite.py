#!/usr/bin/env python3
"""Unified benchmark and comparison suite runner for Broadcast Linux.

Runs identical workloads across:
  - `baseline`: No Broadcast (raw media passthrough & zero-process host baseline)
  - `python`:   Existing Python `nv_broadcast` daemon, DSP, video, and CLI
  - `rust`:     Rust `nvb-*` workspace (Phase R0 -> R4 automatic capability progression)

Usage:
  python3 bench/run_suite.py --targets baseline,python,rust --suites all --profile quick
  python3 bench/run_suite.py --targets baseline,python,rust --suites all --profile full --soak 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.adapters.base import TargetAdapter
from bench.adapters.baseline_adapter import BaselineAdapter
from bench.adapters.python_adapter import PythonAdapter
from bench.adapters.rust_adapter import RustAdapter
from bench.metrics.audio_metrics import build_audio_corpus, compare_target_waveforms, read_wav_mono16
from bench.reports.render_html import render_html
from bench.reports.render_markdown import render_markdown

ALL_SUITES = (
    "audio_quality",
    "audio_latency",
    "memory",
    "video",
    "api_cli",
    "system_impact",
    "reliability",
)


def _git_info(repo_root: Path) -> tuple[str, str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL
        ).strip()
        return branch, sha
    except (OSError, subprocess.SubprocessError):
        return "unknown", "unknown"


def evaluate_gates(suites: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate the acceptance gates defined in `docs/benchmark-suite-plan.md`."""
    gates: list[dict[str, Any]] = []

    def _add(name: str, base: str, py: str, rs: str, rule: str, status: str) -> None:
        gates.append(
            {
                "name": name,
                "baseline": base,
                "python": py,
                "rust": rs,
                "target_rule": rule,
                "status": status,
            }
        )

    # 1. Idle daemon RAM (PSS)
    s3 = suites.get("memory", {})
    b_idle = (((s3.get("baseline") or {}).get("checkpoints") or {}).get("idle_standby") or {}).get("pss_mb")
    p_idle = (((s3.get("python") or {}).get("checkpoints") or {}).get("idle_standby") or {}).get("pss_mb")
    r_idle = (((s3.get("rust") or {}).get("checkpoints") or {}).get("idle_standby") or {}).get("pss_mb")
    r3_st = (s3.get("rust") or {}).get("status", "pending_phase_r2")
    _add(
        "Idle daemon RAM (PSS)",
        f"{b_idle:.1f} MB" if b_idle is not None else "—",
        f"{p_idle:.1f} MB" if p_idle is not None else "—",
        f"{r_idle:.1f} MB" if r_idle is not None else f"`{r3_st}`",
        "Rust <= 25 MB",
        ("pass" if r_idle <= 25.0 else "fail") if r_idle is not None else r3_st,
    )

    # 2. Post-CUDA-call idle RAM (reclamation)
    p_post = (((s3.get("python") or {}).get("checkpoints") or {}).get("post_call_idle") or {}).get("pss_mb")
    r_post = (((s3.get("rust") or {}).get("checkpoints") or {}).get("post_call_idle") or {}).get("pss_mb")
    _add(
        "Post-CUDA-call idle RAM (PSS)",
        "0.0 MB",
        f"{p_post:.1f} MB" if p_post is not None else "—",
        f"{r_post:.1f} MB" if r_post is not None else f"`{r3_st}`",
        "Rust <= 25 MB (100% worker reclaim)",
        ("pass" if r_post <= 25.0 else "fail") if r_post is not None else r3_st,
    )

    # 3. Idle GPU VRAM / P-state
    p_vram = (((s3.get("python") or {}).get("checkpoints") or {}).get("idle_standby") or {}).get("process_vram_mib")
    p_ps = (((s3.get("python") or {}).get("checkpoints") or {}).get("idle_standby") or {}).get("gpu_pstate")
    r_vram = (((s3.get("rust") or {}).get("checkpoints") or {}).get("idle_standby") or {}).get("process_vram_mib")
    _add(
        "Idle GPU VRAM / P-state",
        "0 MiB",
        f"{p_vram:.0f} MiB ({p_ps})" if p_vram is not None else "—",
        f"{r_vram:.0f} MiB" if r_vram is not None else f"`{r3_st}`",
        "0 MiB / P8",
        ("pass" if r_vram == 0.0 else "fail") if r_vram is not None else r3_st,
    )

    # 4. Audio 10 ms frame latency p99 (noise_removal)
    s2 = suites.get("audio_latency", {})
    b_p99 = ((((s2.get("baseline") or {}).get("workloads") or {}).get("noise_removal") or {}).get("latency_us") or {}).get("p99")
    p_p99 = ((((s2.get("python") or {}).get("workloads") or {}).get("noise_removal") or {}).get("latency_us") or {}).get("p99")
    r_p99 = ((((s2.get("rust") or {}).get("workloads") or {}).get("noise_removal") or {}).get("latency_us") or {}).get("p99")
    r2_st = (s2.get("rust") or {}).get("status", "pending_phase_r1")
    _add(
        "Audio 10 ms frame latency (p99)",
        f"{b_p99 / 1000.0:.3f} ms" if b_p99 is not None else "—",
        f"{p_p99 / 1000.0:.3f} ms" if p_p99 is not None else "—",
        f"{r_p99 / 1000.0:.3f} ms" if r_p99 is not None else f"`{r2_st}`",
        "Rust <= 0.500 ms (0 misses > 10 ms)",
        ("pass" if r_p99 <= 500.0 else "fail") if r_p99 is not None else r2_st,
    )

    # 5. Audio fan Voice-SNR gain vs raw
    s1 = suites.get("audio_quality", {})
    p_fan = (((s1.get("python") or {}).get("scenarios") or {}).get("fan") or {}).get("delta_voice_snr_db")
    r_fan = (((s1.get("rust") or {}).get("scenarios") or {}).get("fan") or {}).get("delta_voice_snr_db")
    r1_st = (s1.get("rust") or {}).get("status", "pending_phase_r1")
    _add(
        "Audio `fan` Voice-SNR gain vs raw",
        "0.0 dB",
        f"{p_fan:+.1f} dB" if p_fan is not None else "—",
        f"{r_fan:+.1f} dB" if r_fan is not None else f"`{r1_st}`",
        "Within +/-1.5 dB of Python (or better)",
        ("pass" if (p_fan is not None and r_fan >= p_fan - 1.5) else "fail") if r_fan is not None else r1_st,
    )

    # 6. Pure-noise AGC safety
    p_agc = (((s1.get("python") or {}).get("scenarios") or {}).get("pure_noise") or {}).get("agc_noise_safety_pass")
    r_agc = (((s1.get("rust") or {}).get("scenarios") or {}).get("pure_noise") or {}).get("agc_noise_safety_pass")
    _add(
        "Zero-speech AGC safety (`pure_noise`)",
        "0.0 dB",
        ("SAFE" if p_agc else "UNSAFE") if p_agc is not None else "—",
        ("SAFE" if r_agc else "UNSAFE") if r_agc is not None else f"`{r1_st}`",
        "Never boosts pure noise (<= +0.5 dB)",
        ("pass" if r_agc else "fail") if r_agc is not None else r1_st,
    )

    # 7. CLI cold start
    s5 = suites.get("api_cli", {})
    p_cli = (((s5.get("python") or {}).get("cli_cold_start_ms") or {}).get("version") or {}).get("p50")
    r_cli = (((s5.get("rust") or {}).get("cli_cold_start_ms") or {}).get("version") or {}).get("p50")
    _add(
        "CLI cold start (`--version` p50)",
        "N/A",
        f"{p_cli:.1f} ms" if p_cli is not None else "—",
        f"{r_cli:.1f} ms" if r_cli is not None else "—",
        "Rust <= 25.0 ms",
        ("pass" if r_cli <= 25.0 else "fail") if r_cli is not None else "pending_phase_r2",
    )

    # 8. HTTP /api/state p95 latency
    p_http = ((((s5.get("python") or {}).get("http_endpoints") or {}).get("GET /api/state") or {}).get("latency_ms") or {}).get("p95")
    r_http = ((((s5.get("rust") or {}).get("http_endpoints") or {}).get("GET /api/state") or {}).get("latency_ms") or {}).get("p95")
    _add(
        "HTTP `/api/state` p95 latency",
        "N/A",
        f"{p_http:.2f} ms" if p_http is not None else "—",
        f"{r_http:.2f} ms" if r_http is not None else "`pending_phase_r2`",
        "Rust <= 5.0 ms",
        ("pass" if r_http <= 5.0 else "fail") if r_http is not None else "pending_phase_r2",
    )

    # 9. Idle background wakeups
    s6 = suites.get("system_impact", {})
    p_wake = (((s6.get("python") or {}).get("process_overhead") or {}).get("total_ctxt_per_sec"))
    r_wake = (((s6.get("rust") or {}).get("process_overhead") or {}).get("total_ctxt_per_sec"))
    r6_st = (s6.get("rust") or {}).get("status", "pending_phase_r2")
    _add(
        "Idle background wakeups (ctxt/s)",
        "0.0 /s",
        f"{p_wake:.1f} /s" if p_wake is not None else "—",
        f"{r_wake:.1f} /s" if r_wake is not None else f"`{r6_st}`",
        "Rust <= 50 /s",
        ("pass" if r_wake <= 50.0 else "fail") if r_wake is not None else r6_st,
    )

    # 10. Reliability invariants (active in Phase R0!)
    s7 = suites.get("reliability", {})
    p_rel = s7.get("python") or {}
    r_rel = s7.get("rust") or {}
    p_rel_s = f"{p_rel['passed']}/{p_rel['total']}" if p_rel.get("status") == "ok" else "—"
    r_rel_s = f"{r_rel['passed']}/{r_rel['total']}" if r_rel.get("status") == "ok" else "—"
    rel_pass = (
        "pass"
        if (r_rel.get("status") == "ok" and r_rel.get("passed") == r_rel.get("total"))
        else ("fail" if r_rel.get("status") == "ok" else "pending_phase_r0")
    )
    _add(
        "Reliability & anti-loop invariants",
        "N/A",
        p_rel_s,
        r_rel_s,
        "100% checks passing",
        rel_pass,
    )

    return gates


def compute_cross_target_parity(suites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute automated Python-vs-Rust differential parity where both targets produced outputs."""
    parity: dict[str, Any] = {}

    # Audio waveform parity
    s1 = suites.get("audio_quality", {})
    p_sc = (s1.get("python") or {}).get("scenarios") or {}
    r_sc = (s1.get("rust") or {}).get("scenarios") or {}
    if p_sc and r_sc:
        audio_par: dict[str, Any] = {}
        for name in p_sc:
            if name in r_sc and p_sc[name].get("wav_path") and r_sc[name].get("wav_path"):
                pcm_p, sr = read_wav_mono16(Path(p_sc[name]["wav_path"]))
                pcm_r, _ = read_wav_mono16(Path(r_sc[name]["wav_path"]))
                audio_par[name] = compare_target_waveforms(pcm_p, pcm_r, sample_rate=sr)
        if audio_par:
            parity["audio_waveform"] = audio_par

    # HTTP API contract parity
    s5 = suites.get("api_cli", {})
    p_c = (s5.get("python") or {}).get("contract_snapshot")
    r_c = (s5.get("rust") or {}).get("contract_snapshot")
    if p_c and r_c:
        diffs = {k: {"python": p_c.get(k), "rust": r_c.get(k)} for k in set(p_c) | set(r_c) if p_c.get(k) != r_c.get(k)}
        parity["api_contract"] = {"exact_match": len(diffs) == 0, "diffs": diffs}

    return parity


def build_adapters(
    target_names: list[str],
    repo_root: Path,
    python_bin: str | None = None,
    rust_bin: str | None = None,
    rust_release: bool = False,
) -> list[TargetAdapter]:
    adapters: list[TargetAdapter] = []
    for t in target_names:
        if t == "baseline":
            adapters.append(BaselineAdapter())
        elif t == "python":
            adapters.append(PythonAdapter(repo_root, python_bin=python_bin))
        elif t == "rust":
            ra = RustAdapter(repo_root, rust_cli_bin=rust_bin)
            if rust_release:
                ra.ensure_binary(release=True)
            adapters.append(ra)
        else:
            raise ValueError(f"Unknown target: {t}")
    return adapters


def run_benchmark_suite(
    *,
    targets: list[str],
    selected_suites: list[str],
    profile: str = "quick",
    soak_sec: int = 0,
    custom_wav: Path | None = None,
    live_camera: bool = False,
    out_dir: Path | None = None,
    python_bin: str | None = None,
    rust_bin: str | None = None,
    rust_release: bool = False,
    render_html_report: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, Any], Path]:
    now = dt.datetime.now(dt.timezone.utc)
    if out_dir is None:
        stamp = now.strftime("%Y%m%d-%H%M%S")
        out_dir = REPO_ROOT / "bench" / "results" / f"{stamp}_{profile}"
    out_dir.mkdir(parents=True, exist_ok=True)

    adapters = build_adapters(
        targets,
        REPO_ROOT,
        python_bin=python_bin,
        rust_bin=rust_bin,
        rust_release=rust_release,
    )

    branch, sha = _git_info(REPO_ROOT)
    caps = {ad.name: ad.capabilities() for ad in adapters}
    suites_out: dict[str, dict[str, Any]] = {}

    audio_scenarios = (
        build_audio_corpus(profile=profile, custom_wav=custom_wav)
        if "audio_quality" in selected_suites
        else []
    )

    for sname in selected_suites:
        suites_out[sname] = {}
        for ad in adapters:
            if verbose:
                print(f"[bench] suite={sname} target={ad.name} profile={profile} ...", flush=True)
            try:
                if sname == "audio_quality":
                    res = ad.run_audio_quality(audio_scenarios, out_dir, profile=profile)
                elif sname == "audio_latency":
                    res = ad.run_audio_latency(profile=profile)
                elif sname == "memory":
                    res = ad.run_memory(profile=profile, soak_sec=soak_sec)
                elif sname == "video":
                    res = ad.run_video(profile=profile, out_dir=out_dir, live_camera=live_camera)
                elif sname == "api_cli":
                    res = ad.run_api_cli(profile=profile)
                elif sname == "system_impact":
                    res = ad.run_system_impact(profile=profile)
                elif sname == "reliability":
                    res = ad.run_reliability(profile=profile)
                else:
                    raise ValueError(f"Unknown suite: {sname}")
            except Exception as exc:
                res = {"status": "error", "error": str(exc)}
            suites_out[sname][ad.name] = res

    report: dict[str, Any] = {
        "schema_version": 1,
        "meta": {
            "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "profile": profile,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "git_branch": branch,
            "git_sha": sha,
            "out_dir": str(out_dir),
        },
        "targets": targets,
        "target_capabilities": caps,
        "suites": suites_out,
        "parity": compute_cross_target_parity(suites_out),
        "gates": evaluate_gates(suites_out),
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if render_html_report:
        (out_dir / "report.html").write_text(render_html(report), encoding="utf-8")

    results_root = REPO_ROOT / "bench" / "results"
    if results_root in out_dir.parents:
        (results_root / "latest_run.txt").write_text(str(out_dir) + "\n", encoding="utf-8")

    return report, out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Broadcast Linux unified benchmark & comparison runner")
    p.add_argument("--targets", default="baseline,python,rust", help="Comma-separated targets: baseline,python,rust")
    p.add_argument("--suites", default="all", help=f"Comma-separated suites or 'all': {','.join(ALL_SUITES)}")
    p.add_argument("--profile", choices=("quick", "full"), default="quick", help="Benchmark profile")
    p.add_argument("--soak", type=int, default=0, help="Optional memory soak duration in seconds")
    p.add_argument("--wav", type=str, default=None, help="Optional real voice 16-bit WAV file for audio_quality")
    p.add_argument("--live-camera", action="store_true", help="Include live physical webcam probe if present")
    p.add_argument("--python-bin", type=str, default=None, help="Explicit Python interpreter for PythonAdapter")
    p.add_argument("--rust-bin", type=str, default=None, help="Explicit nvb-cli binary path for RustAdapter")
    p.add_argument("--rust-release", action="store_true", help="Build/prefer release nvb-cli binary")
    p.add_argument("--out-dir", type=str, default=None, help="Custom output directory for report artifacts")
    p.add_argument("--no-html", action="store_true", help="Skip generating report.html")
    p.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if args.suites.strip().lower() == "all":
        selected = list(ALL_SUITES)
    else:
        selected = [s.strip() for s in args.suites.split(",") if s.strip()]
        invalid = [s for s in selected if s not in ALL_SUITES]
        if invalid:
            print(f"error: unknown suites: {', '.join(invalid)}", file=sys.stderr)
            return 2

    report, out_dir = run_benchmark_suite(
        targets=targets,
        selected_suites=selected,
        profile=args.profile,
        soak_sec=args.soak,
        custom_wav=Path(args.wav) if args.wav else None,
        live_camera=args.live_camera,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        python_bin=args.python_bin,
        rust_bin=args.rust_bin,
        rust_release=args.rust_release,
        render_html_report=not args.no_html,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f"\nBenchmark complete -> {out_dir}")
        print(f"  JSON: {out_dir / 'report.json'}")
        print(f"  MD:   {out_dir / 'report.md'}")
        if not args.no_html:
            print(f"  HTML: {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
