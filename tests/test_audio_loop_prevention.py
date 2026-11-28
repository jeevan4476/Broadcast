"""Regression tests for PipeWire audio routing isolation and feedback-loop prevention."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

NV_BCAST_ROOT = Path(__file__).resolve().parent.parent
if str(NV_BCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(NV_BCAST_ROOT))

from nv_broadcast.audio_engine import AudioEngine, _read_chunk_nonblocking
from nv_broadcast.config import BroadcastConfig
from nv_broadcast.devices import _pretty_audio_name
from nv_broadcast.pipewire import (
    VIRTUAL_MIC_SOURCE_NAME,
    VIRTUAL_SPEAKER_SINK_NAME,
    build_isolated_pw_cat_cmd,
    is_pw_link_active,
    is_virtual_broadcast_node,
    link_pw_nodes,
    parse_pw_links,
    select_port_pairs,
)


class TestPipeWireLoopPrevention(unittest.TestCase):
    def test_isolated_pw_cat_cmd_disables_autoconnect_and_reconnect(self) -> None:
        rec = build_isolated_pw_cat_cmd("record", "nvb_spk_cap", rate=48000, channels=2, latency_ms=20)
        self.assertIn("--record", rec)
        self.assertIn("--channels=2", rec)
        props_idx = rec.index("-P") + 1
        self.assertIn('node.name = "nvb_spk_cap"', rec[props_idx])
        self.assertIn("node.autoconnect = false", rec[props_idx])
        self.assertIn("node.dont-reconnect = true", rec[props_idx])

        play = build_isolated_pw_cat_cmd("playback", "nvb_spk_out", rate=48000, channels=2, latency_ms=20)
        self.assertIn("--playback", play)
        play_props = play[play.index("-P") + 1]
        self.assertIn('node.name = "nvb_spk_out"', play_props)
        self.assertIn("node.autoconnect = false", play_props)
        self.assertIn("node.dont-reconnect = true", play_props)

    def test_is_virtual_broadcast_node_matches_virtual_devices_only(self) -> None:
        self.assertTrue(is_virtual_broadcast_node(VIRTUAL_SPEAKER_SINK_NAME))
        self.assertTrue(is_virtual_broadcast_node(f"{VIRTUAL_SPEAKER_SINK_NAME}.monitor"))
        self.assertTrue(is_virtual_broadcast_node(VIRTUAL_MIC_SOURCE_NAME))
        self.assertFalse(is_virtual_broadcast_node("alsa_output.pci-0000_00_1f.3.HiFi__Speaker__sink"))
        self.assertFalse(is_virtual_broadcast_node("alsa_input.pci-0000_00_1f.3.HiFi__Mic1__source"))

    def test_select_port_pairs_stereo_mono_and_aux(self) -> None:
        # Stereo speaker monitor -> stereo capture
        pairs = select_port_pairs(
            VIRTUAL_SPEAKER_SINK_NAME,
            "nvb_spk_cap",
            [f"{VIRTUAL_SPEAKER_SINK_NAME}:monitor_FL", f"{VIRTUAL_SPEAKER_SINK_NAME}:monitor_FR"],
            ["nvb_spk_cap:input_FL", "nvb_spk_cap:input_FR"],
            src_prefix="monitor_",
            dst_prefix="input_",
        )
        self.assertEqual(
            pairs,
            [
                (f"{VIRTUAL_SPEAKER_SINK_NAME}:monitor_FL", "nvb_spk_cap:input_FL"),
                (f"{VIRTUAL_SPEAKER_SINK_NAME}:monitor_FR", "nvb_spk_cap:input_FR"),
            ],
        )

        # Stereo physical mic -> mono capture (uses FL only, avoids +6 dB dual-mono summing)
        mic_pairs = select_port_pairs(
            "alsa_input.mic",
            "nvb_mic_cap",
            ["alsa_input.mic:capture_FL", "alsa_input.mic:capture_FR"],
            ["nvb_mic_cap:input_MONO"],
            src_prefix="capture_",
            dst_prefix="input_",
        )
        self.assertEqual(mic_pairs, [("alsa_input.mic:capture_FL", "nvb_mic_cap:input_MONO")])

        # Aux-labeled USB sink
        aux_pairs = select_port_pairs(
            "nvb_spk_out",
            "usb_dac",
            ["nvb_spk_out:output_FL", "nvb_spk_out:output_FR"],
            ["usb_dac:playback_AUX0", "usb_dac:playback_AUX1"],
            src_prefix="output_",
            dst_prefix="playback_",
        )
        self.assertEqual(
            aux_pairs,
            [
                ("nvb_spk_out:output_FL", "usb_dac:playback_AUX0"),
                ("nvb_spk_out:output_FR", "usb_dac:playback_AUX1"),
            ],
        )

    def test_link_pw_nodes_refuses_virtual_feedback_loops(self) -> None:
        with mock.patch("nv_broadcast.pipewire.shutil.which", return_value="/usr/bin/pw-link"):
            # Refuses self-link
            ok, links = link_pw_nodes("nvb_spk_out", "nvb_spk_out")
            self.assertFalse(ok)
            self.assertEqual(links, [])

            # Refuses linking playback output back into virtual speaker sink
            ok, links = link_pw_nodes(
                "nvb_spk_out",
                VIRTUAL_SPEAKER_SINK_NAME,
                allow_virtual_src=False,
                allow_virtual_dst=False,
            )
            self.assertFalse(ok)
            self.assertEqual(links, [])

            # Refuses linking mic capture from virtual mic or virtual speaker monitor
            ok, links = link_pw_nodes(
                VIRTUAL_MIC_SOURCE_NAME,
                "nvb_mic_cap",
                allow_virtual_src=False,
                allow_virtual_dst=False,
            )
            self.assertFalse(ok)
            self.assertEqual(links, [])

    def test_parse_pw_links_and_detect_dropped_links(self) -> None:
        sample = """
alsa_input.mic:capture_FL
  |-> nvb_mic_cap:input_MONO
nv_broadcast_speakers:monitor_FL
  |-> nvb_spk_cap:input_FL
nvb_spk_out:output_FL
  |-> alsa_output.headphones:playback_FL
"""
        edges = parse_pw_links(sample)
        self.assertTrue(is_pw_link_active("alsa_input.mic", "nvb_mic_cap", links=edges))
        self.assertTrue(is_pw_link_active(VIRTUAL_SPEAKER_SINK_NAME, "nvb_spk_cap", links=edges))
        self.assertTrue(is_pw_link_active("nvb_spk_out", "alsa_output.headphones", links=edges))
        # Missing link (e.g. dropped on suspend/resume) returns False
        self.assertFalse(is_pw_link_active("nvb_mic_feed", VIRTUAL_MIC_SOURCE_NAME, links=edges))
        self.assertFalse(is_pw_link_active("nvb_spk_out", "alsa_output.speakers", links=edges))

    def test_nonblocking_chunk_read_timeout_partial_and_eof(self) -> None:
        r_fd, w_fd = os.pipe()
        try:
            buf = bytearray()
            # No data available -> returns timeout promptly instead of blocking forever
            chunk, timed_out = _read_chunk_nonblocking(r_fd, buf, 8, timeout=0.05)
            self.assertIsNone(chunk)
            self.assertTrue(timed_out)

            # Partial write -> times out without losing buffered bytes, then completes
            os.write(w_fd, b"abcd")
            chunk, timed_out = _read_chunk_nonblocking(r_fd, buf, 8, timeout=0.05)
            self.assertIsNone(chunk)
            self.assertTrue(timed_out)
            self.assertEqual(bytes(buf), b"abcd")

            os.write(w_fd, b"efgh12")
            chunk, timed_out = _read_chunk_nonblocking(r_fd, buf, 8, timeout=0.2)
            self.assertFalse(timed_out)
            self.assertEqual(chunk, b"abcdefgh")
            self.assertEqual(bytes(buf), b"")
            self.assertEqual(os.read(r_fd, 2), b"12")

            # Close writer -> EOF
            os.close(w_fd)
            w_fd = -1
            chunk, timed_out = _read_chunk_nonblocking(r_fd, buf, 8, timeout=0.2)
            self.assertFalse(timed_out)
            self.assertEqual(chunk, b"")
        finally:
            os.close(r_fd)
            if w_fd != -1:
                os.close(w_fd)

    def test_play_wav_targets_physical_speaker_not_virtual_default(self) -> None:
        with TemporaryDirectory() as tmp:
            cfg = BroadcastConfig(Path(tmp) / "config.json")
            wav = Path(tmp) / "demo.wav"
            wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            engine = AudioEngine(cfg)
            engine.resolve_physical_speaker = mock.Mock(return_value="alsa_output.headphones")
            fake_proc = mock.Mock()
            fake_proc.poll.return_value = 0
            with (
                mock.patch("nv_broadcast.audio_engine.shutil.which", return_value="/usr/bin/pw-play"),
                mock.patch("nv_broadcast.audio_engine.subprocess.Popen", return_value=fake_proc) as popen_mock,
            ):
                ok, _ = engine.play_wav(str(wav))
            self.assertTrue(ok)
            cmd = popen_mock.call_args.args[0]
            self.assertIn("--target=alsa_output.headphones", cmd)

    def test_pretty_audio_name_and_lazy_probe_avoids_gpu_imports(self) -> None:
        self.assertEqual(
            _pretty_audio_name("alsa_output.pci-0000_00_1f.3.HiFi__Headphones__sink"),
            "Wired Headphones (3.5mm)",
        )
        # Verify importing server + probing capabilities in a clean interpreter does NOT load CUDA/MediaPipe/cv2
        code = (
            "import sys; "
            "from nv_broadcast.config import BroadcastConfig; "
            "from nv_broadcast.devices import probe_capabilities; "
            "from nv_broadcast.video_engine import VideoEngine; "
            "probe_capabilities(); "
            "ve = VideoEngine(BroadcastConfig('/tmp/nvb_lazy_test_cfg.json')); "
            "assert ve.segmentation_backend == 'standby'; "
            "assert ve.face_backend == 'standby'; "
            "loaded = [m for m in ('onnxruntime', 'mediapipe', 'cv2') if m in sys.modules]; "
            "assert not loaded, f'Heavy modules loaded eagerly: {loaded}'"
        )
        env = {**os.environ, "PYTHONPATH": str(NV_BCAST_ROOT)}
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
