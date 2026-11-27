"""Regression tests for PipeWire audio routing isolation and feedback-loop prevention."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

NV_BCAST_ROOT = Path(__file__).resolve().parent.parent
if str(NV_BCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(NV_BCAST_ROOT))

from nv_broadcast.pipewire import (
    VIRTUAL_MIC_SOURCE_NAME,
    VIRTUAL_SPEAKER_SINK_NAME,
    build_isolated_pw_cat_cmd,
    is_virtual_broadcast_node,
    link_pw_nodes,
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


if __name__ == "__main__":
    unittest.main()
