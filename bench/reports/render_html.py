"""Render a self-contained dark/light HTML benchmark dashboard with inline SVG charts."""

from __future__ import annotations

import html
from typing import Any


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _svg_grouped_bars(
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, str, list[float | None]]],
    unit: str = "",
    reference_line: tuple[float, str] | None = None,
) -> str:
    """Build a crisp responsive inline SVG grouped bar chart."""
    w = 680
    pad_l, pad_r, pad_t, pad_b = 58, 20, 44, 56
    h = 250
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    all_vals = [v for _, _, vals in series for v in vals if v is not None]
    if reference_line is not None:
        all_vals.append(reference_line[0])
    max_val = max(all_vals) if all_vals else 1.0
    min_val = min(0.0, min(all_vals)) if all_vals else 0.0
    span = max(max_val - min_val, 1e-6)
    max_val = max_val + span * 0.14
    if min_val < 0:
        min_val = min_val - span * 0.10
    span = max(max_val - min_val, 1e-6)

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1.0 - (v - min_val) / span)

    zero_y = y_of(0.0)
    parts: list[str] = [
        f'<div class="chart-card">',
        f'  <div class="chart-head"><div><h3>{_esc(title)}</h3><p>{_esc(subtitle)}</p></div>',
        '  <div class="legend">',
    ]
    for sname, scolor, _ in series:
        parts.append(f'<span class="leg-item"><i style="background:{scolor}"></i>{_esc(sname)}</span>')
    parts.append("</div></div>")
    parts.append(f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{_esc(title)}">')

    # Gridlines
    for gi in range(5):
        gv = min_val + span * (gi / 4.0)
        gy = y_of(gv)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" class="grid-line"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" class="axis-lbl">{gv:.1f}{_esc(unit)}</text>')

    parts.append(f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{w - pad_r}" y2="{zero_y:.1f}" class="zero-line"/>')

    if reference_line is not None:
        ry = y_of(reference_line[0])
        parts.append(
            f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{w - pad_r}" y2="{ry:.1f}" class="ref-line"/>'
            f'<text x="{w - pad_r - 4}" y="{ry - 5:.1f}" text-anchor="end" class="ref-lbl">{_esc(reference_line[1])}</text>'
        )

    n_cat = max(1, len(categories))
    n_ser = max(1, len(series))
    group_w = plot_w / n_cat
    bar_w = min(28.0, (group_w * 0.72) / n_ser)

    for ci, cat in enumerate(categories):
        gx = pad_l + ci * group_w + group_w * 0.5
        total_bw = bar_w * n_ser + (n_ser - 1) * 3.0
        start_x = gx - total_bw * 0.5
        for si, (_, scolor, vals) in enumerate(series):
            val = vals[ci] if ci < len(vals) else None
            bx = start_x + si * (bar_w + 3.0)
            if val is None:
                continue
            y_val = y_of(val)
            top = min(zero_y, y_val)
            bh = max(2.0, abs(zero_y - y_val))
            parts.append(
                f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" fill="{scolor}">'
                f"<title>{_esc(cat)}: {val:.2f}{_esc(unit)}</title></rect>"
            )
            lbl_y = top - 4.0 if val >= 0 else top + bh + 11.0
            parts.append(
                f'<text x="{bx + bar_w * 0.5:.1f}" y="{lbl_y:.1f}" text-anchor="middle" class="bar-val">{val:.1f}</text>'
            )
        short_cat = cat if len(cat) <= 15 else cat[:14] + "…"
        parts.append(
            f'<text x="{gx:.1f}" y="{h - 18}" text-anchor="middle" class="cat-lbl">{_esc(short_cat)}</text>'
        )

    parts.append("</svg></div>")
    return "\n".join(parts)


def render_html(report: dict[str, Any]) -> str:
    meta = report.get("meta", {})
    suites = report.get("suites", {})
    gates = report.get("gates", [])

    # KPI values
    s1 = suites.get("audio_quality", {})
    s2 = suites.get("audio_latency", {})
    s3 = suites.get("memory", {})
    s4 = suites.get("video", {})
    s5 = suites.get("api_cli", {})
    s7 = suites.get("reliability", {})

    py_snr_gain = ((s1.get("python") or {}).get("summary") or {}).get("mean_delta_si_sdr_db")
    py_vsnr_vals = [
        v["delta_voice_snr_db"]
        for v in ((s1.get("python") or {}).get("scenarios") or {}).values()
        if "delta_voice_snr_db" in v and "delta_snr_db" in v
    ]
    py_vsnr_gain = (sum(py_vsnr_vals) / len(py_vsnr_vals)) if py_vsnr_vals else py_snr_gain
    py_idle_pss = (((s3.get("python") or {}).get("checkpoints") or {}).get("idle_standby") or {}).get("pss_mb")
    py_post_pss = (((s3.get("python") or {}).get("checkpoints") or {}).get("post_call_idle") or {}).get("pss_mb")
    py_cfg_ops = (((s5.get("python") or {}).get("config_bench")) or {}).get("ops_per_sec")
    rs_cfg_ops = (((s5.get("rust") or {}).get("config_bench")) or {}).get("ops_per_sec")
    cfg_speedup = (rs_cfg_ops / py_cfg_ops) if (py_cfg_ops and rs_cfg_ops) else None

    py_pcm_p50 = ((((s2.get("python") or {}).get("pcm_primitive")) or {}).get("latency_us") or {}).get("p50")
    rs_pcm_p50 = ((((s2.get("rust") or {}).get("pcm_primitive")) or {}).get("latency_us") or {}).get("p50")
    pcm_speedup = (py_pcm_p50 / max(rs_pcm_p50, 1e-6)) if (py_pcm_p50 and rs_pcm_p50) else None

    charts_html: list[str] = []

    # Chart 1: Audio Quality Gain (Delta-SNR & Delta-SI-SDR vs Baseline)
    py_sc = (s1.get("python") or {}).get("scenarios", {})
    audio_cats = [k for k, v in py_sc.items() if "delta_snr_db" in v]
    if audio_cats:
        charts_html.append(
            _svg_grouped_bars(
                "Audio Enhancement Gain vs No Broadcast (Baseline = 0 dB)",
                "Positive dB = cleaner speech over raw noisy microphone audio",
                audio_cats,
                [
                    ("Delta-SI-SDR", "#76b900", [py_sc[k]["delta_si_sdr_db"] for k in audio_cats]),
                    ("Voice-SNR Gain", "#38bdf8", [py_sc[k].get("delta_voice_snr_db", 0.0) for k in audio_cats]),
                    ("Silence Floor Cut", "#a78bfa", [py_sc[k]["silence_attenuation_db"] for k in audio_cats]),
                ],
                unit=" dB",
            )
        )

    # Chart 2: Memory Lifecycle (PSS MB across 5 states)
    py_cps = (s3.get("python") or {}).get("checkpoints", {})
    rs_cps = (s3.get("rust") or {}).get("checkpoints", {})
    cp_keys = [k for k in ("idle_standby", "active_audio_call", "active_video_cpu", "active_video_cuda", "post_call_idle") if k in py_cps]
    if cp_keys:
        mem_series: list[tuple[str, str, list[float | None]]] = [
            ("Baseline (0 MB)", "#64748b", [0.0 for _ in cp_keys]),
            ("Python PSS (MB)", "#f59e0b", [py_cps[k]["pss_mb"] for k in cp_keys]),
            ("Python RSS (MB)", "#ef4444", [py_cps[k]["rss_mb"] for k in cp_keys]),
        ]
        if rs_cps:
            mem_series.append(("Rust PSS (MB)", "#76b900", [rs_cps.get(k, {}).get("pss_mb") for k in cp_keys]))
        charts_html.append(
            _svg_grouped_bars(
                "Process-Tree Memory Lifecycle (PSS / RSS MB)",
                "Exposes Python post-CUDA sticky RAM vs Rust <= 25 MB idle target",
                [k.replace("_", " ") for k in cp_keys],
                mem_series,
                unit=" MB",
                reference_line=(25.0, "Rust idle target (25 MB)"),
            )
        )

    # Chart 3: Video FPS across effects
    py_ve = (s4.get("python") or {}).get("effects", {})
    if py_ve:
        v_cats = list(py_ve.keys())
        charts_html.append(
            _svg_grouped_bars(
                f"Video Pipeline Sustained FPS ({(s4.get('python') or {}).get('resolution', '')}, {(s4.get('python') or {}).get('backend', '')})",
                "30 FPS real-time webcam budget reference line",
                [k.replace("background_", "bg_").replace("combo_blur_keylight_autoframe", "combo_3fx") for k in v_cats],
                [("Python FPS", "#38bdf8", [py_ve[k]["fps"] for k in v_cats])],
                unit=" fps",
                reference_line=(30.0, "30 FPS budget"),
            )
        )

    # Chart 4: Phase R0 Rust vs Python Live Speedup (microseconds / ms)
    r0_cats: list[str] = []
    r0_py: list[float | None] = []
    r0_rs: list[float | None] = []
    py_cfg_p50 = ((s5.get("python") or {}).get("config_bench") or {}).get("latency", {}).get("p50_us")
    rs_cfg_p50 = ((s5.get("rust") or {}).get("config_bench") or {}).get("latency", {}).get("p50_us")
    if py_cfg_p50 and rs_cfg_p50:
        r0_cats.append("Config toggle (us)")
        r0_py.append(py_cfg_p50)
        r0_rs.append(rs_cfg_p50)
    if py_pcm_p50 and rs_pcm_p50:
        r0_cats.append("PCM16+RMS (us)")
        r0_py.append(py_pcm_p50)
        r0_rs.append(rs_pcm_p50)
    py_ver = ((s5.get("python") or {}).get("cli_cold_start_ms") or {}).get("version", {}).get("p50")
    rs_ver = ((s5.get("rust") or {}).get("cli_cold_start_ms") or {}).get("version", {}).get("p50")
    if py_ver and rs_ver:
        r0_cats.append("CLI --version (ms)")
        r0_py.append(py_ver)
        r0_rs.append(rs_ver)
    if r0_cats:
        charts_html.append(
            _svg_grouped_bars(
                "Live Python vs Rust Latency (Lower is Better)",
                "Measured on identical workloads (Phase R0 active Rust primitives)",
                r0_cats,
                [
                    ("Python", "#f59e0b", r0_py),
                    ("Rust", "#76b900", r0_rs),
                ],
            )
        )

    # Acceptance gate table rows
    gate_rows = []
    for g in gates:
        st = g["status"]
        cls = "badge-pass" if st == "pass" else ("badge-fail" if st == "fail" else "badge-pend")
        gate_rows.append(
            f"<tr><td>{_esc(g['name'])}</td><td class='num'>{_esc(g.get('baseline', '—'))}</td>"
            f"<td class='num'>{_esc(g.get('python', '—'))}</td><td class='num'>{_esc(g.get('rust', '—'))}</td>"
            f"<td>{_esc(g.get('target_rule', ''))}</td><td><span class='badge {cls}'>{_esc(st.upper())}</span></td></tr>"
        )

    # Reliability rows
    p_rel = {c["name"]: c for c in (s7.get("python") or {}).get("checks", [])}
    r_rel = {c["name"]: c for c in (s7.get("rust") or {}).get("checks", [])}
    rel_rows = []
    for cname in dict.fromkeys([*p_rel.keys(), *r_rel.keys()]):
        pc = p_rel.get(cname)
        rc = r_rel.get(cname)
        p_badge = "<span class='badge badge-pass'>PASS</span>" if (pc and pc["passed"]) else ("<span class='badge badge-fail'>FAIL</span>" if pc else "—")
        r_badge = "<span class='badge badge-pass'>PASS</span>" if (rc and rc["passed"]) else ("<span class='badge badge-fail'>FAIL</span>" if rc else "—")
        detail = (rc or pc or {}).get("detail", "")
        rel_rows.append(f"<tr><td><code>{_esc(cname)}</code></td><td>{p_badge}</td><td>{r_badge}</td><td>{_esc(detail)}</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Broadcast Linux — Unified Benchmark Report</title>
<style>
  :root {{
    --bg: #0f1318; --card: #161c24; --border: #27313f; --text: #e6edf5; --muted: #94a3b8;
    --accent: #76b900; --pass: #22c55e; --fail: #ef4444; --pend: #eab308;
  }}
  [data-theme="light"] {{
    --bg: #f8fafc; --card: #ffffff; --border: #dbe3ee; --text: #0f172a; --muted: #475569;
    --accent: #5c9400; --pass: #15803d; --fail: #dc2626; --pend: #b45309;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, sans-serif; line-height: 1.5; }}
  .wrap {{ max-width: 1120px; margin: 0 auto; }}
  header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  h1 {{ margin: 0 0 6px; font-size: 1.5rem; }}
  .meta {{ color: var(--muted); font-size: 0.875rem; }}
  .theme-btn {{ background: var(--card); color: var(--text); border: 1px solid var(--border); padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .kpi .lbl {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
  .kpi .val {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; color: var(--accent); }}
  .kpi .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 2px; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-bottom: 22px; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .chart-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }}
  .chart-head h3 {{ margin: 0; font-size: 1rem; }}
  .chart-head p {{ margin: 2px 0 0; font-size: 0.8rem; color: var(--muted); }}
  .legend {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.78rem; color: var(--muted); }}
  .leg-item {{ display: inline-flex; align-items: center; gap: 5px; }}
  .leg-item i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .grid-line {{ stroke: var(--border); stroke-width: 1; stroke-dasharray: 2 3; }}
  .zero-line {{ stroke: var(--muted); stroke-width: 1; }}
  .ref-line {{ stroke: #76b900; stroke-width: 1.5; stroke-dasharray: 5 4; }}
  .axis-lbl, .cat-lbl {{ fill: var(--muted); font-size: 11px; }}
  .bar-val {{ fill: var(--text); font-size: 10px; font-weight: 600; }}
  .ref-lbl {{ fill: #76b900; font-size: 11px; font-weight: 600; }}
  .panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 18px; overflow-x: auto; }}
  .panel h2 {{ margin: 0 0 12px; font-size: 1.1rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.74rem; font-weight: 700; }}
  .badge-pass {{ background: rgba(34,197,94,.16); color: var(--pass); }}
  .badge-fail {{ background: rgba(239,68,68,.16); color: var(--fail); }}
  .badge-pend {{ background: rgba(234,179,8,.16); color: var(--pend); }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Broadcast Linux — Unified Benchmark Dashboard</h1>
      <div class="meta">
        Profile: <strong>{_esc(meta.get('profile', 'quick'))}</strong> ·
        Generated: {_esc(meta.get('timestamp_utc', ''))} ·
        Git: <code>{_esc(meta.get('git_branch', ''))}@{_esc(meta.get('git_sha', ''))}</code> ·
        Targets: {_esc(', '.join(report.get('targets', [])))}
      </div>
    </div>
    <button class="theme-btn" onclick="document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'">Toggle Dark / Light</button>
  </header>

  <section class="kpis">
    <div class="kpi">
      <div class="lbl">Audio Voice-SNR Gain (Python vs Raw)</div>
      <div class="val">{f'{py_vsnr_gain:+.1f} dB' if py_vsnr_gain is not None else '—'}</div>
      <div class="sub">Mean speech-to-silence SNR gain across noisy corpus</div>
    </div>
    <div class="kpi">
      <div class="lbl">Python Idle vs Post-CUDA PSS</div>
      <div class="val">{f'{py_idle_pss:.0f} → {py_post_pss:.0f} MB' if (py_idle_pss is not None and py_post_pss is not None) else '—'}</div>
      <div class="sub">Rust target: ≤ 25 MB before &amp; after calls</div>
    </div>
    <div class="kpi">
      <div class="lbl">Phase R0 Config Engine Speedup</div>
      <div class="val">{f'{cfg_speedup:.1f}x' if cfg_speedup is not None else '—'}</div>
      <div class="sub">Rust vs Python sanitize + conflict toggle</div>
    </div>
    <div class="kpi">
      <div class="lbl">Phase R0 PCM16 + RMS Speedup</div>
      <div class="val">{f'{pcm_speedup:.1f}x' if pcm_speedup is not None else '—'}</div>
      <div class="sub">480-sample 48 kHz audio primitive</div>
    </div>
  </section>

  <section class="charts">
    {''.join(charts_html)}
  </section>

  <section class="panel">
    <h2>Target Acceptance Gates</h2>
    <table>
      <thead><tr><th>Gate</th><th style="text-align:right">Baseline</th><th style="text-align:right">Python</th><th style="text-align:right">Rust</th><th>Target</th><th>Status</th></tr></thead>
      <tbody>{''.join(gate_rows)}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Suite 7: Reliability &amp; Anti-Loop Chaos Matrix</h2>
    <table>
      <thead><tr><th>Check</th><th>Python</th><th>Rust</th><th>Invariant Verified</th></tr></thead>
      <tbody>{''.join(rel_rows)}</tbody>
    </table>
  </section>
</div>
</body>
</html>
"""
