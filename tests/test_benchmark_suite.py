"""Unit and integration tests for the unified `bench/` benchmark suite."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from bench.compare import compare_reports, format_diff_markdown
from bench.metrics.audio_metrics import (
    SAMPLE_RATE,
    build_audio_corpus,
    compare_target_waveforms,
    compute_seg_snr_db,
    compute_si_sdr_db,
    compute_snr_db,
    evaluate_audio_clip,
    float_list_to_pcm16,
    read_wav_mono16,
    write_wav_mono16,
)
from bench.metrics.system_metrics import (
    compute_percentiles,
    parse_proc_stat_cpu_ticks,
    parse_proc_status,
    parse_pw_top_batch,
    parse_smaps_rollup,
)
from bench.metrics.video_metrics import (
    compute_mask_iou_mae,
    compute_psnr_db,
    compute_ssim,
    generate_synthetic_portrait,
)
from bench.reports.render_html import render_html
from bench.reports.render_markdown import render_markdown
from bench.run_suite import evaluate_gates, run_benchmark_suite


class TestAudioMetrics(unittest.TestCase):
    def test_snr_and_si_sdr_handle_gain_and_causal_lag(self) -> None:
        # 0.2s harmonic speech-like tone
        n = 9600
        clean = [0.4 * math.sin(2.0 * math.pi * 220.0 * i / SAMPLE_RATE) for i in range(n)]
        # Delayed by 120 samples (2.5ms) and scaled by 1.5x (e.g. AGC)
        delayed_scaled = ([0.0] * 120) + [1.5 * x for x in clean[:-120]]
        snr_aligned, lag = compute_snr_db(clean, delayed_scaled, align=True)
        sisdr_aligned = compute_si_sdr_db(clean, delayed_scaled, align=True)
        self.assertGreater(abs(lag), 100)
        self.assertGreater(sisdr_aligned, 30.0)
        # SNR is scale-sensitive (1.5x gain has 0.5x amplitude error -> ~6 dB), SI-SDR is gain-invariant
        self.assertGreater(snr_aligned, 4.0)

    def test_wav_roundtrip_and_corpus_evaluation(self) -> None:
        corpus = build_audio_corpus(profile="quick")
        names = [s.name for s in corpus]
        self.assertIn("fan", names)
        self.assertIn("pure_noise", names)
        fan = next(s for s in corpus if s.name == "fan")

        with tempfile.TemporaryDirectory() as td:
            wav_p = write_wav_mono16(Path(td) / "fan.wav", fan.noisy_pcm, fan.sample_rate)
            raw_back, sr = read_wav_mono16(wav_p)
            self.assertEqual(sr, SAMPLE_RATE)
            self.assertEqual(raw_back, fan.noisy_pcm)

        # Passthrough must yield ~0 dB Delta-SNR and 0 dB silence attenuation
        ev_raw = evaluate_audio_clip(fan.noisy_pcm, fan.noisy_pcm, fan.clean_pcm, fan.speech_mask)
        self.assertAlmostEqual(ev_raw["delta_snr_db"], 0.0, places=2)
        self.assertAlmostEqual(ev_raw["silence_attenuation_db"], 0.0, places=2)

        # Clean output must yield high positive Delta-SNR and Delta-SI-SDR
        assert fan.clean_pcm is not None
        ev_clean = evaluate_audio_clip(fan.noisy_pcm, fan.clean_pcm, fan.clean_pcm, fan.speech_mask)
        self.assertGreater(ev_clean["delta_snr_db"], 20.0)
        self.assertGreater(ev_clean["delta_si_sdr_db"], 20.0)

        # Waveform parity against self is near 1.0 correlation and 0 dB RMS delta
        par = compare_target_waveforms(fan.clean_pcm, fan.clean_pcm)
        self.assertAlmostEqual(par["rms_delta_db"], 0.0, places=2)
        self.assertGreater(par["max_cross_correlation"], 0.99)

        # Segmental SNR sanity
        clean_f = [0.3 * math.sin(2.0 * math.pi * 300.0 * i / SAMPLE_RATE) for i in range(4800)]
        seg = compute_seg_snr_db(clean_f, clean_f, [True] * 10)
        self.assertGreater(seg, 30.0)

    def test_pure_noise_agc_safety_flag(self) -> None:
        quiet_noise = float_list_to_pcm16([0.05 * math.sin(i * 0.1) for i in range(4800)])
        boosted_noise = float_list_to_pcm16([0.25 * math.sin(i * 0.1) for i in range(4800)])
        ev_safe = evaluate_audio_clip(quiet_noise, quiet_noise, None, [False] * 10)
        ev_bad = evaluate_audio_clip(quiet_noise, boosted_noise, None, [False] * 10)
        self.assertTrue(ev_safe["agc_noise_safety_pass"])
        self.assertFalse(ev_bad["agc_noise_safety_pass"])


class TestVideoAndSystemMetrics(unittest.TestCase):
    def test_synthetic_portrait_and_image_metrics(self) -> None:
        frame, gt = generate_synthetic_portrait(160, 90, frame_index=0)
        frame2, _ = generate_synthetic_portrait(160, 90, frame_index=4)
        self.assertGreater(compute_psnr_db(frame, frame), 90.0)
        self.assertAlmostEqual(compute_ssim(frame, frame), 1.0, places=3)
        self.assertLess(compute_psnr_db(frame, frame2), 60.0)
        iou_mae = compute_mask_iou_mae(gt, gt)
        self.assertEqual(iou_mae["iou"], 1.0)
        self.assertEqual(iou_mae["mae"], 0.0)

    def test_proc_and_pw_top_parsers(self) -> None:
        smaps = "Rss: 102400 kB\nPss: 81920 kB\nPrivate_Dirty: 65536 kB\nSwapPss: 0 kB\n"
        sm = parse_smaps_rollup(smaps)
        self.assertEqual(sm["rss_kb"], 102400)
        self.assertEqual(sm["pss_kb"], 81920)
        self.assertEqual(sm["private_dirty_kb"], 65536)

        status = "Name:\tpython3\nThreads:\t7\nVmRSS:\t90000 kB\nvoluntary_ctxt_switches:\t1200\nnonvoluntary_ctxt_switches:\t45\n"
        st = parse_proc_status(status)
        self.assertEqual(st["threads"], 7)
        self.assertEqual(st["voluntary_ctxt_switches"], 1200)
        self.assertEqual(st["nonvoluntary_ctxt_switches"], 45)

        stat_line = "1234 (python 3.12) S 1 1234 1234 0 -1 4194304 100 0 0 0 42 17 0 0 20 0 1 0 1000"
        ut, kt = parse_proc_stat_cpu_ticks(stat_line)
        self.assertEqual((ut, kt), (42, 17))

        pw_sample = (
            "S   ID  QUANT   RATE    WAIT    BUSY   W/Q   B/Q  ERR FORMAT           NAME\n"
            "R   42    480  48000  20.0us  15.0us  0.00  0.00    0 F32LE 2 48000    nv_broadcast_mic_capture\n"
            "S   55      0      0    ---     ---   ---   ---     1                  alsa_output.pci\n"
        )
        rows = parse_pw_top_batch(pw_sample)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["name"], "F32LE 2 48000 nv_broadcast_mic_capture")
        self.assertEqual(rows[1]["err"], 1)

        pct = compute_percentiles([10.0, 20.0, 30.0, 40.0, 100.0])
        self.assertEqual(pct["min"], 10.0)
        self.assertEqual(pct["p50"], 30.0)
        self.assertEqual(pct["max"], 100.0)


class TestRunnerReportsAndCompare(unittest.TestCase):
    def test_compare_reports_detects_regressions(self) -> None:
        before = {
            "meta": {"git_sha": "aaa"},
            "targets": ["baseline", "python", "rust"],
            "suites": {
                "memory": {
                    "rust": {"status": "ok", "checkpoints": {"idle_standby": {"pss_mb": 18.0}}},
                },
                "audio_quality": {
                    "rust": {"status": "ok", "scenarios": {"fan": {"delta_snr_db": 14.0}}},
                },
                "reliability": {
                    "rust": {"status": "ok", "passed": 5, "total": 5},
                },
            },
        }
        after_ok = {
            "meta": {"git_sha": "bbb"},
            "targets": ["baseline", "python", "rust"],
            "suites": {
                "memory": {
                    "rust": {"status": "ok", "checkpoints": {"idle_standby": {"pss_mb": 19.0}}},
                },
                "audio_quality": {
                    "rust": {"status": "ok", "scenarios": {"fan": {"delta_snr_db": 14.5}}},
                },
                "reliability": {
                    "rust": {"status": "ok", "passed": 5, "total": 5},
                },
            },
        }
        after_bad = {
            "meta": {"git_sha": "ccc"},
            "targets": ["baseline", "python", "rust"],
            "suites": {
                "memory": {
                    "rust": {"status": "ok", "checkpoints": {"idle_standby": {"pss_mb": 30.0}}},
                },
                "audio_quality": {
                    "rust": {"status": "ok", "scenarios": {"fan": {"delta_snr_db": 10.0}}},
                },
                "reliability": {
                    "rust": {"status": "ok", "passed": 4, "total": 5},
                },
            },
        }
        diff_ok = compare_reports(before, after_ok)
        self.assertTrue(diff_ok["ok"])
        self.assertEqual(len(diff_ok["regressions"]), 0)

        diff_bad = compare_reports(before, after_bad)
        self.assertFalse(diff_bad["ok"])
        self.assertEqual(len(diff_bad["regressions"]), 3)
        md = format_diff_markdown(diff_bad)
        self.assertIn("REGRESSION", md)

    def test_end_to_end_quick_smoke_across_baseline_python_rust(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "smoke_run"
            report, written_dir = run_benchmark_suite(
                targets=["baseline", "python", "rust"],
                selected_suites=["reliability", "audio_quality"],
                profile="quick",
                out_dir=out_dir,
                python_bin=sys.executable,
                render_html_report=True,
                verbose=False,
            )
            self.assertEqual(written_dir, out_dir)
            self.assertTrue((out_dir / "report.json").is_file())
            self.assertTrue((out_dir / "report.md").is_file())
            self.assertTrue((out_dir / "report.html").is_file())

            # Verify JSON roundtrips and reports baseline, python, and rust cleanly
            loaded = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["targets"], ["baseline", "python", "rust"])
            self.assertEqual(loaded["suites"]["reliability"]["python"]["status"], "ok")
            self.assertEqual(
                loaded["suites"]["reliability"]["python"]["passed"],
                loaded["suites"]["reliability"]["python"]["total"],
            )
            # Audio quality runs on baseline and python, and cleanly reports pending_phase_r1 (or ok) on rust
            self.assertEqual(loaded["suites"]["audio_quality"]["baseline"]["status"], "ok")
            self.assertEqual(loaded["suites"]["audio_quality"]["python"]["status"], "ok")
            self.assertIn(
                loaded["suites"]["audio_quality"]["rust"]["status"],
                ("ok", "pending_phase_r1", "unavailable"),
            )

            md = render_markdown(report)
            self.assertIn("Suite 1: Audio Signal Quality", md)
            self.assertIn("Suite 7: Reliability", md)
            html_out = render_html(report)
            self.assertIn("<svg", html_out)
            self.assertIn("Broadcast Linux", html_out)
            self.assertGreater(len(evaluate_gates(report["suites"])), 5)


if __name__ == "__main__":
    unittest.main()
