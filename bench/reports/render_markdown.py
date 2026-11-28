"""Render side-by-side Markdown comparison reports from benchmark `report.json`."""

from __future__ import annotations

from typing import Any


def _fmt_num(val: Any, unit: str = "", digits: int = 2, signed: bool = False) -> str:
    if val is None or isinstance(val, bool):
        return "N/A"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return str(val)
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{fmt.format(num)}{(' ' + unit) if unit else ''}"


def _status_badge(status: str) -> str:
    if status == "pass":
        return "PASS"
    if status == "fail":
        return "FAIL"
    if status.startswith("pending"):
        return f"PENDING ({status.replace('pending_', '').upper()})"
    return status.upper()


def _cell_for_target(suite_data: dict[str, Any] | None, getter: Any, fallback_pending: bool = True) -> str:
    if not suite_data:
        return "—"
    st = str(suite_data.get("status", ""))
    if st in ("not_applicable", "unavailable", "error"):
        return "N/A" if st == "not_applicable" else st
    if st.startswith("pending") and fallback_pending:
        return f"`{st}`"
    try:
        val = getter(suite_data)
    except (KeyError, TypeError, IndexError):
        val = None
    if val is None:
        return f"`{st}`" if st.startswith("pending") else "—"
    return str(val)


def render_markdown(report: dict[str, Any]) -> str:
    meta = report.get("meta", {})
    targets: list[str] = report.get("targets", [])
    suites: dict[str, dict[str, Any]] = report.get("suites", {})
    gates: list[dict[str, Any]] = report.get("gates", [])

    lines: list[str] = []
    lines.append("# Broadcast Linux — Unified Benchmark & Comparison Report")
    lines.append("")
    lines.append(
        f"- **Generated:** `{meta.get('timestamp_utc', 'unknown')}` | "
        f"**Profile:** `{meta.get('profile', 'quick')}` | "
        f"**Git:** `{meta.get('git_branch', 'unknown')}@{meta.get('git_sha', 'unknown')}`"
    )
    lines.append(
        f"- **Host:** `{meta.get('hostname', 'unknown')}` | "
        f"**Targets:** {', '.join(f'`{t}`' for t in targets)}"
    )
    py_caps = report.get("target_capabilities", {}).get("python", {})
    rs_caps = report.get("target_capabilities", {}).get("rust", {})
    if py_caps:
        lines.append(
            f"- **Python Runtime:** `{py_caps.get('python_bin', 'unknown')}` "
            f"(audio backend: `{py_caps.get('audio_backend', 'unknown')}`)"
        )
    if rs_caps:
        lines.append(
            f"- **Rust Binary:** `{rs_caps.get('binary', 'unbuilt')}` "
            f"(phase: `{rs_caps.get('phase', 'unknown')}`)"
        )
    lines.append("")

    # Acceptance Gates
    if gates:
        lines.append("## Target Acceptance Gates")
        lines.append("")
        lines.append("| Gate | Baseline | Python | Rust | Target | Status |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for g in gates:
            lines.append(
                f"| {g['name']} | {g.get('baseline', '—')} | {g.get('python', '—')} | "
                f"{g.get('rust', '—')} | {g.get('target_rule', '')} | **{_status_badge(g['status'])}** |"
            )
        lines.append("")

    # Suite 1: Audio Signal Quality
    if "audio_quality" in suites:
        s1 = suites["audio_quality"]
        lines.append("## Suite 1: Audio Signal Quality & Noise Reduction (`audio_quality`)")
        lines.append("")
        base_sc = (s1.get("baseline") or {}).get("scenarios", {})
        py_sc = (s1.get("python") or {}).get("scenarios", {})
        rs_sc = (s1.get("rust") or {}).get("scenarios", {})
        scenario_names = list(base_sc.keys() or py_sc.keys() or rs_sc.keys())
        rs_status = (s1.get("rust") or {}).get("status", "—")

        lines.append(
            "| Scenario | Baseline SNR / Voice-SNR | Python Voice-SNR Gain (Sil Atten) | "
            "Rust Voice-SNR Gain (Sil Atten) | Python vs Rust Parity |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        audio_parity = (report.get("parity") or {}).get("audio_waveform") or {}
        for name in scenario_names:
            b = base_sc.get(name, {})
            p = py_sc.get(name, {})
            r = rs_sc.get(name, {})
            if "output_snr_db" in b or "output_snr_db" in p or "output_snr_db" in r:
                b_col = f"{_fmt_num(b.get('output_snr_db'), 'dB', 1)} / {_fmt_num(b.get('output_voice_snr_db'), 'dB', 1)}"
                p_col = (
                    f"{_fmt_num(p.get('delta_voice_snr_db'), 'dB', 1, signed=True)} "
                    f"({_fmt_num(p.get('silence_attenuation_db'), 'dB', 1, signed=True)})"
                    if p
                    else "—"
                )
                r_col = (
                    f"{_fmt_num(r.get('delta_voice_snr_db'), 'dB', 1, signed=True)} "
                    f"({_fmt_num(r.get('silence_attenuation_db'), 'dB', 1, signed=True)})"
                    if r
                    else (f"`{rs_status}`" if "rust" in targets else "—")
                )
            else:
                b_col = f"RMS {_fmt_num(b.get('output_rms_db'), 'dB', 1)}"
                p_safe = "SAFE" if p.get("agc_noise_safety_pass") else "BOOST!"
                p_col = (
                    f"{_fmt_num(p.get('rms_delta_db'), 'dB', 1, signed=True)} ({p_safe})"
                    if p
                    else "—"
                )
                if r:
                    r_safe = "SAFE" if r.get("agc_noise_safety_pass") else "BOOST!"
                    r_col = f"{_fmt_num(r.get('rms_delta_db'), 'dB', 1, signed=True)} ({r_safe})"
                else:
                    r_col = f"`{rs_status}`" if "rust" in targets else "—"
            par = audio_parity.get(name)
            par_col = (
                f"dRMS {par['rms_delta_db']:+.2f} dB, corr {par['max_cross_correlation']:.3f}"
                if par
                else "—"
            )
            lines.append(f"| `{name}` | {b_col} | {p_col} | {r_col} | {par_col} |")
        if (s1.get("rust") or {}).get("peak_rss_mb"):
            lines.append("")
            lines.append(f"> **Rust `nvb-audio` peak RSS during WAV processing:** `{s1['rust']['peak_rss_mb']:.2f} MB` (`{s1['rust'].get('backend', 'nnnoiseless')}`)")
        lines.append("")

    # Suite 2: Real-Time Audio Latency
    if "audio_latency" in suites:
        s2 = suites["audio_latency"]
        lines.append("## Suite 2: Real-Time Audio Latency & Jitter (`audio_latency`, 10 ms budget)")
        lines.append("")
        lines.append("| Workload | Metric | Baseline | Python | Rust |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for wname in ("noise_removal", "room_echo_removal", "studio_voice"):
            b_w = ((s2.get("baseline") or {}).get("workloads") or {}).get(wname)
            p_w = ((s2.get("python") or {}).get("workloads") or {}).get(wname)
            r_w = ((s2.get("rust") or {}).get("workloads") or {}).get(wname)
            r_st = (s2.get("rust") or {}).get("status", "—")
            for metric_label, fn in (
                ("p50 / p99 / max (us)", lambda w: f"{w['latency_us']['p50']:.1f} / {w['latency_us']['p99']:.1f} / {w['latency_us']['max']:.1f}"),
                ("RTF (1x CPU%)", lambda w: f"{w['rtf']:.4f} ({w['single_core_cpu_pct']:.2f}%)"),
                ("Misses >2ms / >10ms", lambda w: f"{w['budget_misses_2ms']} / {w['budget_misses_10ms']}"),
            ):
                b_c = fn(b_w) if b_w else "—"
                p_c = fn(p_w) if p_w else "—"
                r_c = fn(r_w) if r_w else (f"`{r_st}`" if "rust" in targets else "—")
                lines.append(f"| `{wname}` | {metric_label} | {b_c} | {p_c} | {r_c} |")

        # Phase R0 PCM16 + RMS primitive comparison (live Rust vs Python!)
        py_pcm = (s2.get("python") or {}).get("pcm_primitive")
        rs_pcm = (s2.get("rust") or {}).get("pcm_primitive")
        if py_pcm or rs_pcm:
            lines.append("")
            lines.append("### Phase R0 Live Primitive: 480-sample PCM16 <-> Float + RMS (`pcm_primitive`)")
            lines.append("")
            lines.append("| Target | p50 (us) | p95 (us) | p99 (us) | max (us) | Total Wall (ms) | Speedup vs Python |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            py_p50 = (py_pcm or {}).get("latency_us", {}).get("p50", 0.0)
            for tname, pdata in (("python", py_pcm), ("rust", rs_pcm)):
                if not pdata:
                    continue
                lu = pdata["latency_us"]
                sp = f"{py_p50 / max(lu['p50'], 1e-6):.1f}x" if (tname == "rust" and py_p50 > 0) else "1.0x"
                lines.append(
                    f"| `{tname}` | {lu['p50']:.2f} | {lu['p95']:.2f} | {lu['p99']:.2f} | "
                    f"{lu['max']:.2f} | {pdata['wall_ms']:.2f} | **{sp}** |"
                )
        lines.append("")

    # Suite 3: Memory & VRAM Lifecycle
    if "memory" in suites:
        s3 = suites["memory"]
        lines.append("## Suite 3: Process-Tree RAM/PSS & GPU VRAM Lifecycle (`memory`)")
        lines.append("")
        lines.append("| Lifecycle Checkpoint | Baseline (PSS / VRAM) | Python (PSS / RSS / VRAM) | Rust (PSS / VRAM) |")
        lines.append("| --- | ---: | ---: | ---: |")
        r_st = (s3.get("rust") or {}).get("status", "—")
        for cp in ("idle_standby", "active_audio_call", "active_video_cpu", "active_video_cuda", "post_call_idle"):
            b = ((s3.get("baseline") or {}).get("checkpoints") or {}).get(cp)
            p = ((s3.get("python") or {}).get("checkpoints") or {}).get(cp)
            r = ((s3.get("rust") or {}).get("checkpoints") or {}).get(cp)
            b_c = f"{b['pss_mb']:.1f} MB / {b['process_vram_mib']:.0f} MiB" if b else "—"
            p_c = (
                f"{p['pss_mb']:.1f} MB / {p['rss_mb']:.1f} MB / {p['process_vram_mib']:.0f} MiB"
                if p
                else "—"
            )
            r_c = (
                f"{r['pss_mb']:.1f} MB / {r['process_vram_mib']:.0f} MiB"
                if r
                else (f"`{r_st}`" if "rust" in targets else "—")
            )
            lines.append(f"| `{cp}` | {b_c} | {p_c} | {r_c} |")
        live = (s3.get("python") or {}).get("live_systemd_service")
        if live:
            lines.append("")
            lines.append(
                f"> **Live `nv-broadcast.service` snapshot (read-only):** "
                f"{live['process_count']} procs, **{live['pss_mb']:.1f} MB PSS** "
                f"({live['rss_mb']:.1f} MB RSS), {live['threads']} threads, "
                f"{live['open_fds']} open FDs, {live['process_vram_mib']:.0f} MiB VRAM (`{live['gpu_pstate']}`)."
            )
        lines.append("")

    # Suite 4: Video Quality & FPS
    if "video" in suites:
        s4 = suites["video"]
        lines.append("## Suite 4: Video Quality, FPS & Frame Jitter (`video`)")
        lines.append("")
        p_v = s4.get("python") or {}
        b_v = s4.get("baseline") or {}
        r_v = s4.get("rust") or {}
        r_st = r_v.get("status", "—")
        if p_v.get("status") == "ok":
            lines.append(
                f"- **Python Video Backends:** segmentation=`{p_v.get('backend')}`, "
                f"face=`{p_v.get('face_backend')}`, resolution=`{p_v.get('resolution')}`"
            )
            if p_v.get("mask_vs_synthetic_gt"):
                mm = p_v["mask_vs_synthetic_gt"]
                lines.append(f"- **Python Matte vs Synthetic GT:** IoU=`{mm['iou']}`, MAE=`{mm['mae']}`")
        if r_v.get("status") == "ok":
            lines.append(
                f"- **Rust Video Backends:** segmentation=`{r_v.get('backend')}`, "
                f"face=`{r_v.get('face_backend')}`, resolution=`{r_v.get('resolution')}`, "
                f"worker peak RSS=`{r_v.get('peak_rss_mb', 0):.1f} MB`"
            )
            if r_v.get("mask_vs_synthetic_gt"):
                rmm = r_v["mask_vs_synthetic_gt"]
                lines.append(f"- **Rust Matte vs Synthetic GT:** IoU=`{rmm['iou']}`, MAE=`{rmm['mae']}`")
        if p_v.get("status") == "ok" or r_v.get("status") == "ok":
            lines.append("")
        lines.append("| Effect | Baseline FPS | Python FPS (p50/p95 ms) | Rust FPS (p50/p95 ms) | Speedup | Rust Quality |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        effect_names = list((p_v.get("effects") or r_v.get("effects") or b_v.get("effects") or {}).keys())
        for ename in effect_names:
            be = (b_v.get("effects") or {}).get(ename)
            pe = (p_v.get("effects") or {}).get(ename)
            re_ = (r_v.get("effects") or {}).get(ename)
            b_fps = f"{be['fps']:.1f}" if be else "—"
            p_col = (
                f"{pe['fps']:.1f} ({pe['latency_ms']['p50']:.1f}/{pe['latency_ms']['p95']:.1f})"
                if pe
                else "—"
            )
            if re_:
                r_col = f"{re_['fps']:.1f} ({re_['latency_ms']['p50']:.1f}/{re_['latency_ms']['p95']:.1f})"
                sp = f"**{re_['fps'] / max(pe['fps'], 0.01):.2f}x**" if pe else "—"
                if "denoise_psnr_gain_db" in re_:
                    q_col = f"+{re_['denoise_psnr_gain_db']:.2f} dB PSNR"
                else:
                    q_col = f"{re_['bg_smoothness_delta_pct']:+.1f}% grad"
            else:
                r_col = f"`{r_st}`" if "rust" in targets else "—"
                sp = "—"
                q_col = "—"
            lines.append(f"| `{ename}` | {b_fps} | {p_col} | {r_col} | {sp} | {q_col} |")
        lines.append("")

    # Suite 5: Control Plane, CLI & HTTP/SSE API
    if "api_cli" in suites:
        s5 = suites["api_cli"]
        p5 = s5.get("python") or {}
        r5 = s5.get("rust") or {}
        lines.append("## Suite 5: Control Plane, CLI Startup & HTTP/SSE API (`api_cli`)")
        lines.append("")
        lines.append("| Metric | Python | Rust | Speedup / Delta |")
        lines.append("| --- | ---: | ---: | ---: |")

        p_cfg = p5.get("config_bench") or {}
        r_cfg = r5.get("config_bench") or {}
        if p_cfg.get("status") == "ok" or r_cfg.get("status") == "ok":
            p_ops = p_cfg.get("ops_per_sec")
            r_ops = r_cfg.get("ops_per_sec")
            sp = f"**{r_ops / p_ops:.1f}x faster**" if (p_ops and r_ops) else "—"
            lines.append(
                f"| Config sanitize + conflict toggle throughput | "
                f"{_fmt_num(p_ops, 'ops/s', 0)} | {_fmt_num(r_ops, 'ops/s', 0)} | {sp} |"
            )
            p_p50 = (p_cfg.get("latency") or {}).get("p50_us")
            r_p50 = (r_cfg.get("latency") or {}).get("p50_us")
            sp_l = f"**{p_p50 / max(r_p50, 1e-6):.1f}x lower**" if (p_p50 and r_p50) else "—"
            lines.append(
                f"| Config toggle `p50` latency | {_fmt_num(p_p50, 'us', 1)} | {_fmt_num(r_p50, 'us', 1)} | {sp_l} |"
            )

        p_ver = (p5.get("cli_cold_start_ms") or {}).get("version", {}).get("p50")
        r_ver = (r5.get("cli_cold_start_ms") or {}).get("version", {}).get("p50")
        if p_ver or r_ver:
            sp_v = f"**{p_ver / max(r_ver, 1e-6):.1f}x faster**" if (p_ver and r_ver) else "—"
            lines.append(
                f"| CLI cold start `--version` (`p50`) | {_fmt_num(p_ver, 'ms', 1)} | {_fmt_num(r_ver, 'ms', 1)} | {sp_v} |"
            )

        for cname in ("status", "devices", "effects", "doctor"):
            cv = (p5.get("cli_cold_start_ms") or {}).get(cname)
            rv = (r5.get("cli_cold_start_ms") or {}).get(cname)
            if cv:
                r_txt = f"{rv['p50']:.1f} / {rv['p95']:.1f} ms" if rv else "`pending_phase_r2`"
                sp_c = f"**{cv['p50'] / max(rv['p50'], 1e-6):.1f}x faster**" if rv else "—"
                lines.append(
                    f"| CLI cold start `{cname}` (`p50` / `p95`) | "
                    f"{cv['p50']:.1f} / {cv['p95']:.1f} ms | {r_txt} | {sp_c} |"
                )

        if p5.get("daemon_boot_ms") is not None:
            rb = r5.get("daemon_boot_ms")
            r_boot = _fmt_num(rb, "ms", 1) if rb is not None else "`pending_phase_r2`"
            sp_b = f"**{p5['daemon_boot_ms'] / max(rb, 1e-6):.1f}x faster**" if rb else "—"
            lines.append(f"| Daemon cold boot to `READY` | {p5['daemon_boot_ms']:.1f} ms | {r_boot} | {sp_b} |")

        for ep, ep_data in (p5.get("http_endpoints") or {}).items():
            lm = ep_data["latency_ms"]
            r_ep = (r5.get("http_endpoints") or {}).get(ep)
            if r_ep:
                rlm = r_ep["latency_ms"]
                r_txt = f"{rlm['p50']:.2f} / {rlm['p95']:.2f} ms ({r_ep['req_per_sec']:.0f} req/s)"
                sp_e = f"**{lm['p95'] / max(rlm['p95'], 1e-6):.1f}x lower p95**"
            else:
                r_txt = "`pending_phase_r2`"
                sp_e = "—"
            lines.append(
                f"| HTTP `{ep}` (`p50` / `p95` ms, req/s) | "
                f"{lm['p50']:.2f} / {lm['p95']:.2f} ms ({ep_data['req_per_sec']:.0f} req/s) | {r_txt} | {sp_e} |"
            )

        for skey, sdata in (p5.get("sse_fanout") or {}).items():
            rs_sse = (r5.get("sse_fanout") or {}).get(skey)
            r_txt = (
                f"{rs_sse['cpu_ms_per_sec']:.1f} CPU-ms/s, {rs_sse['total_ctxt_per_sec']:.1f} ctxt/s, {rs_sse['rss_mb']:.1f} MB"
                if rs_sse
                else "`pending_phase_r2`"
            )
            lines.append(
                f"| SSE `/api/events` ({skey}) | "
                f"{sdata['cpu_ms_per_sec']:.1f} CPU-ms/s, {sdata['total_ctxt_per_sec']:.1f} ctxt/s, {sdata['rss_mb']:.1f} MB | {r_txt} | — |"
            )

        api_par = (report.get("parity") or {}).get("api_contract")
        if api_par is not None:
            lines.append("")
            if api_par.get("exact_match"):
                lines.append("> **Differential API Contract Parity (Python vs Rust):** **EXACT MATCH** (status codes, auth rejection, JSON keys, and effect conflict results identical).")
            else:
                lines.append(f"> **Differential API Contract Parity (Python vs Rust):** DIFFS FOUND: `{api_par.get('diffs')}`")
        lines.append("")

    # Suite 6: System, Power & Idle Wakeup Overhead
    if "system_impact" in suites:
        s6 = suites["system_impact"]
        lines.append("## Suite 6: Background System, Power & Idle Wakeup Overhead (`system_impact`)")
        lines.append("")
        lines.append("| Metric | Baseline (No Broadcast) | Python (Idle Daemon) | Rust |")
        lines.append("| --- | ---: | ---: | ---: |")
        b6 = s6.get("baseline") or {}
        p6 = s6.get("python") or {}
        r6 = s6.get("rust") or {}
        r_st = r6.get("status", "—")
        rows = [
            ("Process tree PSS (MB)", lambda d: _fmt_num(d["process_overhead"]["pss_mb"], "MB", 1)),
            ("CPU time (ms / wall-sec)", lambda d: _fmt_num(d["process_overhead"]["cpu_ms_per_sec"], "ms/s", 2)),
            ("CPU (% of 1 core)", lambda d: _fmt_num(d["process_overhead"]["cpu_core_pct"], "%", 2)),
            ("Threads / Open FDs", lambda d: f"{d['process_overhead']['threads']} / {d['process_overhead']['open_fds']}"),
            ("Context switches / sec (wakeups)", lambda d: _fmt_num(d["process_overhead"]["total_ctxt_per_sec"], "/s", 1)),
            ("GPU P-state / Power (W)", lambda d: f"`{d['gpu'].get('pstate', 'N/A')}` / {_fmt_num(d['gpu'].get('power_draw_w'), 'W', 2)}"),
            ("Process GPU VRAM (MiB)", lambda d: _fmt_num(d["gpu"].get("process_vram_mib"), "MiB", 0)),
        ]
        for label, fn in rows:
            bc = _cell_for_target(b6, fn)
            pc = _cell_for_target(p6, fn)
            rc = _cell_for_target(r6, fn) if "rust" in targets else "—"
            if rc == "—" and "rust" in targets:
                rc = f"`{r_st}`"
            lines.append(f"| {label} | {bc} | {pc} | {rc} |")
        lines.append("")

    # Suite 7: Reliability & Anti-Loop Matrix
    if "reliability" in suites:
        s7 = suites["reliability"]
        lines.append("## Suite 7: Reliability, Anti-Loop & Recovery Chaos Matrix (`reliability`)")
        lines.append("")
        p_checks = {c["name"]: c for c in (s7.get("python") or {}).get("checks", [])}
        r_checks = {c["name"]: c for c in (s7.get("rust") or {}).get("checks", [])}
        all_names = list(dict.fromkeys([*p_checks.keys(), *r_checks.keys()]))
        lines.append("| Check | Baseline | Python | Rust |")
        lines.append("| --- | --- | --- | --- |")
        for cname in all_names:
            pc = p_checks.get(cname)
            rc = r_checks.get(cname)
            p_txt = ("PASS" if pc["passed"] else "FAIL") if pc else "—"
            r_txt = ("PASS" if rc["passed"] else "FAIL") if rc else "—"
            lines.append(f"| `{cname}` | N/A | **{p_txt}** | **{r_txt}** |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
