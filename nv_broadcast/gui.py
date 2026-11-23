"""Native GTK4 + libadwaita Desktop UI styled after NVIDIA Broadcast."""

from __future__ import annotations

import threading
from typing import Any

from . import __app_name__, __version__
from .audio_dsp import AudioMeterReading
from .audio_engine import AudioEngine, MicTestResult
from .config import BG_REPLACE_PRESETS, BLUR_MODES, STUDIO_VOICE_PROFILES, BroadcastConfig
from .devices import list_audio_devices, list_cameras, probe_capabilities
from .v4l2_ctl import list_v4l2_controls, set_v4l2_control
from .video_engine import VideoEngine


NV_CSS = """
window, .background {
    background-color: #121417;
    color: #eef1f5;
}
headerbar {
    background-color: #181b20;
    border-bottom: 1px solid #262b33;
}
.nv-card {
    background-color: #1b1f25;
    border: 1px solid #2a303a;
    border-radius: 10px;
    padding: 14px;
}
.nv-accent-btn {
    background-color: #76b900;
    color: #0b0d0f;
    font-weight: 700;
    border-radius: 6px;
    padding: 8px 16px;
}
.nv-accent-btn:hover {
    background-color: #89d400;
}
.nv-stop-btn {
    background-color: #c93b3b;
    color: #ffffff;
    font-weight: 700;
    border-radius: 6px;
    padding: 8px 16px;
}
.nv-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
}
.nv-subtitle {
    font-size: 12px;
    color: #9aa4b2;
}
.nv-status-bar {
    background-color: #15181d;
    border-top: 1px solid #262b33;
    padding: 8px 14px;
    font-size: 12px;
    color: #b7c0cc;
}
.nv-badge-ok {
    color: #76b900;
    font-weight: 700;
}
.nv-badge-warn {
    color: #f0b429;
    font-weight: 700;
}
progressbar > trough > progress {
    background-color: #76b900;
}
"""


def launch_gui(config: BroadcastConfig | None = None, initial_tab: str = "mic") -> int:
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        from gi.repository import Adw, Gdk, GLib, Gtk  # type: ignore
    except Exception as exc:
        print(f"GTK4 / libadwaita unavailable: {exc}")
        print("Use the CLI (`nv-broadcast doctor`, `nv-broadcast start --mic`) or install python3-gobject gtk4 libadwaita.")
        return 1

    cfg = config or BroadcastConfig()
    audio_engine = AudioEngine(cfg)
    video_engine = VideoEngine(cfg)

    class BroadcastWindow(Adw.ApplicationWindow):
        def __init__(self, app: Adw.Application) -> None:
            super().__init__(application=app, title=f"{__app_name__} v{__version__}")
            self.set_default_size(1020, 720)
            self._updating_ui = False
            self._last_mic_test: MicTestResult | None = None

            provider = Gtk.CssProvider()
            provider.load_from_string(NV_CSS)
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(),
                provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )

            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
            self.set_content(root)

            # Header bar with tab switcher
            header = Adw.HeaderBar()
            title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            logo_lbl = Gtk.Label(label="NVIDIA BROADCAST")
            logo_lbl.add_css_class("nv-badge-ok")
            ver_lbl = Gtk.Label(label="Linux")
            ver_lbl.add_css_class("nv-subtitle")
            title_box.append(logo_lbl)
            title_box.append(ver_lbl)
            header.pack_start(title_box)

            self.stack = Adw.ViewStack()
            switcher = Adw.ViewSwitcher()
            switcher.set_stack(self.stack)
            switcher.set_policy(Adw.ViewSwitcherPolicy.WIDE)
            header.set_title_widget(switcher)
            root.append(header)

            # Build 3 main tabs matching NVIDIA Broadcast
            self.stack.add_titled_with_icon(self._build_mic_tab(), "mic", "Microphone", "audio-input-microphone-symbolic")
            self.stack.add_titled_with_icon(self._build_speaker_tab(), "speakers", "Speakers", "audio-speakers-symbolic")
            self.stack.add_titled_with_icon(self._build_camera_tab(), "camera", "Camera", "camera-web-symbolic")
            if initial_tab in ("mic", "speakers", "camera"):
                self.stack.set_visible_child_name(initial_tab)
            self.stack.set_vexpand(True)
            root.append(self.stack)

            # Bottom Hardware / Doctor status bar
            root.append(self._build_status_bar())

            audio_engine.add_meter_callback(self._on_audio_meter)
            video_engine.add_frame_callback(self._on_video_frame)
            self.connect("close-request", self._on_close)

        # ---- Microphone Tab ----
        def _build_mic_tab(self) -> Gtk.Widget:
            scroller = Gtk.ScrolledWindow()
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
            box.set_margin_top(16)
            box.set_margin_bottom(16)
            box.set_margin_start(18)
            box.set_margin_end(18)
            scroller.set_child(box)

            # Source device + Start/Stop + Live VU Meter card
            top_card = self._make_card()
            top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            lbl = Gtk.Label(label="Microphone Source")
            lbl.add_css_class("nv-title")
            lbl.set_halign(Gtk.Align.START)
            lbl.set_hexpand(True)
            top_row.append(lbl)

            self.mic_toggle_btn = Gtk.Button(label="Start Virtual Mic")
            self.mic_toggle_btn.add_css_class("nv-accent-btn")
            self.mic_toggle_btn.connect("clicked", self._on_toggle_mic_engine)
            top_row.append(self.mic_toggle_btn)
            top_card.append(top_row)

            audio_devs = list_audio_devices()
            self._mic_ids = ["auto"] + [d.name for d in audio_devs["microphones"]]
            mic_labels = ["Auto (System Default)"] + [d.description for d in audio_devs["microphones"]]
            self.mic_dropdown = Gtk.DropDown.new_from_strings(mic_labels)
            cur_mic = cfg.data["microphone"].get("device", "auto")
            if cur_mic in self._mic_ids:
                self.mic_dropdown.set_selected(self._mic_ids.index(cur_mic))
            self.mic_dropdown.connect("notify::selected", self._on_mic_device_changed)
            top_card.append(self.mic_dropdown)

            meter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            meter_lbl = Gtk.Label(label="Input / Output Level")
            meter_lbl.add_css_class("nv-subtitle")
            meter_row.append(meter_lbl)
            self.vu_bar = Gtk.ProgressBar()
            self.vu_bar.set_hexpand(True)
            self.vu_bar.set_valign(Gtk.Align.CENTER)
            meter_row.append(self.vu_bar)
            self.vu_db_lbl = Gtk.Label(label="-90.0 dBFS")
            self.vu_db_lbl.add_css_class("nv-subtitle")
            meter_row.append(self.vu_db_lbl)
            top_card.append(meter_row)
            box.append(top_card)

            # Effects card
            eff_card = self._make_card()
            eff_hdr = Gtk.Label(label="AI Audio Effects")
            eff_hdr.add_css_class("nv-title")
            eff_hdr.set_halign(Gtk.Align.START)
            eff_card.append(eff_hdr)

            mic_effs = cfg.data["microphone"]["effects"]
            self.sw_nr, self.sl_nr = self._add_toggle_slider_row(
                eff_card,
                "Noise Removal",
                "Removes laptop fan, keyboard, and ambient background noise",
                mic_effs["noise_removal"]["enabled"],
                mic_effs["noise_removal"]["strength"],
                lambda en, val: self._set_mic_effect("noise_removal", en, {"strength": val}),
            )
            self.sw_echo, self.sl_echo = self._add_toggle_slider_row(
                eff_card,
                "Room Echo Removal",
                "Suppresses hollow room reflections and reverberation",
                mic_effs["room_echo_removal"]["enabled"],
                mic_effs["room_echo_removal"]["strength"],
                lambda en, val: self._set_mic_effect("room_echo_removal", en, {"strength": val}),
            )

            # Studio Voice row
            sv_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            sv_text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            sv_text.set_hexpand(True)
            sv_t = Gtk.Label(label="Studio Voice")
            sv_t.set_halign(Gtk.Align.START)
            sv_s = Gtk.Label(label="Broadcast vocal EQ & leveling (mutually exclusive with Noise/Echo)")
            sv_s.add_css_class("nv-subtitle")
            sv_s.set_halign(Gtk.Align.START)
            sv_text.append(sv_t)
            sv_text.append(sv_s)
            sv_row.append(sv_text)

            self.sv_profile_dd = Gtk.DropDown.new_from_strings([p.title() for p in STUDIO_VOICE_PROFILES])
            cur_prof = mic_effs["studio_voice"].get("micProfile", "default")
            if cur_prof in STUDIO_VOICE_PROFILES:
                self.sv_profile_dd.set_selected(STUDIO_VOICE_PROFILES.index(cur_prof))
            self.sv_profile_dd.connect("notify::selected", self._on_sv_changed)
            sv_row.append(self.sv_profile_dd)

            self.sw_sv = Gtk.Switch(active=bool(mic_effs["studio_voice"]["enabled"]))
            self.sw_sv.set_valign(Gtk.Align.CENTER)
            self.sw_sv.connect("state-set", self._on_sv_toggled)
            sv_row.append(self.sw_sv)
            eff_card.append(sv_row)

            # Laptop Mic Fixes (HPF + Noise Gate + Auto Gain)
            extras = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
            extras.set_margin_top(6)
            self.chk_hpf = Gtk.CheckButton(label="Fan Rumble Filter (90 Hz HPF)")
            self.chk_hpf.set_active(bool(cfg.data["microphone"].get("highpass_enabled", True)))
            self.chk_hpf.connect("toggled", lambda b: self._set_mic_flag("highpass_enabled", b.get_active()))
            self.chk_gate = Gtk.CheckButton(label="Noise Gate")
            self.chk_gate.set_active(bool(cfg.data["microphone"].get("noise_gate_enabled", True)))
            self.chk_gate.connect("toggled", lambda b: self._set_mic_flag("noise_gate_enabled", b.get_active()))
            self.chk_agc = Gtk.CheckButton(label="Voice Auto-Gain / Compressor")
            self.chk_agc.set_active(bool(cfg.data["microphone"].get("auto_gain_enabled", True)))
            self.chk_agc.connect("toggled", lambda b: self._set_mic_flag("auto_gain_enabled", b.get_active()))
            extras.append(self.chk_hpf)
            extras.append(self.chk_gate)
            extras.append(self.chk_agc)
            eff_card.append(extras)
            box.append(eff_card)

            # Mic Test card (5s A/B comparison)
            test_card = self._make_card()
            test_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            tlbl = Gtk.Label(label="Microphone Test (5s A/B Comparison)")
            tlbl.add_css_class("nv-title")
            tlbl.set_hexpand(True)
            tlbl.set_halign(Gtk.Align.START)
            test_row.append(tlbl)

            self.btn_rec_test = Gtk.Button(label="Record 5s")
            self.btn_rec_test.connect("clicked", self._on_record_mic_test)
            self.btn_play_raw = Gtk.Button(label="Play Raw")
            self.btn_play_raw.set_sensitive(False)
            self.btn_play_raw.connect("clicked", lambda _: self._play_test(False))
            self.btn_play_proc = Gtk.Button(label="Play Processed")
            self.btn_play_proc.set_sensitive(False)
            self.btn_play_proc.connect("clicked", lambda _: self._play_test(True))
            test_row.append(self.btn_rec_test)
            test_row.append(self.btn_play_raw)
            test_row.append(self.btn_play_proc)
            test_card.append(test_row)

            self.test_status_lbl = Gtk.Label(label="Record a 5-second sample to compare raw vs processed microphone audio.")
            self.test_status_lbl.add_css_class("nv-subtitle")
            self.test_status_lbl.set_halign(Gtk.Align.START)
            test_card.append(self.test_status_lbl)
            box.append(test_card)

            return scroller

        # ---- Speakers Tab ----
        def _build_speaker_tab(self) -> Gtk.Widget:
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
            box.set_margin_top(16)
            box.set_margin_bottom(16)
            box.set_margin_start(18)
            box.set_margin_end(18)

            top_card = self._make_card()
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            lbl = Gtk.Label(label="Speaker Output")
            lbl.add_css_class("nv-title")
            lbl.set_hexpand(True)
            lbl.set_halign(Gtk.Align.START)
            row.append(lbl)

            self.spk_toggle_btn = Gtk.Button(label="Start Virtual Speakers")
            self.spk_toggle_btn.add_css_class("nv-accent-btn")
            self.spk_toggle_btn.connect("clicked", self._on_toggle_spk_engine)
            row.append(self.spk_toggle_btn)
            top_card.append(row)

            audio_devs = list_audio_devices()
            self._spk_ids = ["auto"] + [d.name for d in audio_devs["speakers"]]
            spk_labels = ["Auto (System Default)"] + [d.description for d in audio_devs["speakers"]]
            self.spk_dropdown = Gtk.DropDown.new_from_strings(spk_labels)
            cur_spk = cfg.data["speaker"].get("device", "auto")
            if cur_spk in self._spk_ids:
                self.spk_dropdown.set_selected(self._spk_ids.index(cur_spk))
            self.spk_dropdown.connect("notify::selected", self._on_spk_device_changed)
            top_card.append(self.spk_dropdown)
            box.append(top_card)

            eff_card = self._make_card()
            spk_effs = cfg.data["speaker"]["effects"]
            self._add_toggle_slider_row(
                eff_card,
                "Speaker Noise Removal",
                "Removes background noise from incoming Discord/Meet/Zoom audio",
                spk_effs["speaker_noise_removal"]["enabled"],
                spk_effs["speaker_noise_removal"]["strength"],
                lambda en, val: self._set_generic_effect("speaker_noise_removal", en, {"strength": val}),
            )
            self._add_toggle_slider_row(
                eff_card,
                "Speaker Room Echo Removal",
                "Reduces echo on incoming call voices",
                spk_effs["speaker_echo_removal"]["enabled"],
                spk_effs["speaker_echo_removal"]["strength"],
                lambda en, val: self._set_generic_effect("speaker_echo_removal", en, {"strength": val}),
            )
            box.append(eff_card)
            return box

        # ---- Camera Tab ----
        def _build_camera_tab(self) -> Gtk.Widget:
            split = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
            split.set_margin_top(14)
            split.set_margin_bottom(14)
            split.set_margin_start(16)
            split.set_margin_end(16)

            # Left: Live preview + device controls
            left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            left.set_hexpand(True)

            preview_card = self._make_card()
            hdr = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            plbl = Gtk.Label(label="Camera Preview")
            plbl.add_css_class("nv-title")
            plbl.set_hexpand(True)
            plbl.set_halign(Gtk.Align.START)
            hdr.append(plbl)

            self.cam_fps_lbl = Gtk.Label(label="Idle")
            self.cam_fps_lbl.add_css_class("nv-subtitle")
            hdr.append(self.cam_fps_lbl)

            self.cam_toggle_btn = Gtk.Button(label="Start Camera")
            self.cam_toggle_btn.add_css_class("nv-accent-btn")
            self.cam_toggle_btn.connect("clicked", self._on_toggle_cam_engine)
            hdr.append(self.cam_toggle_btn)
            preview_card.append(hdr)

            cams = list_cameras()
            self._cam_paths = ["auto"] + [c.path for c in cams]
            cam_labels = ["Auto (First Physical Webcam)"] + [f"{c.card} ({c.path})" for c in cams]
            self.cam_dropdown = Gtk.DropDown.new_from_strings(cam_labels)
            self.cam_dropdown.connect("notify::selected", self._on_cam_device_changed)
            preview_card.append(self.cam_dropdown)

            self.preview_picture = Gtk.Picture()
            self.preview_picture.set_size_request(480, 270)
            self.preview_picture.set_vexpand(True)
            self.preview_picture.set_hexpand(True)
            preview_card.append(self.preview_picture)

            caps = probe_capabilities()
            if not caps.v4l2loopback_loaded:
                hint = Gtk.Label(
                    label="v4l2loopback not loaded — preview works now; run ./install-deps.sh to expose 'NVIDIA Broadcast Camera' to OBS/Meet/Zoom."
                )
                hint.add_css_class("nv-badge-warn")
                hint.set_wrap(True)
                hint.set_halign(Gtk.Align.START)
                preview_card.append(hint)

            left.append(preview_card)
            split.append(left)

            # Right: Scrollable Camera Effects + V4L2 Hardware Controls
            right_scroll = Gtk.ScrolledWindow()
            right_scroll.set_min_content_width(390)
            right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            right_scroll.set_child(right)

            ceff = cfg.data["camera"]["effects"]
            fx_card = self._make_card()
            fxlbl = Gtk.Label(label="Camera Effects")
            fxlbl.add_css_class("nv-title")
            fxlbl.set_halign(Gtk.Align.START)
            fx_card.append(fxlbl)

            self.sw_blur, _ = self._add_toggle_slider_row(
                fx_card,
                "Background Blur",
                "Softens background behind you",
                ceff["background_blur"]["enabled"],
                ceff["background_blur"]["strength"],
                lambda en, val: self._set_cam_bg_effect("background_blur", en, {"strength": val}),
            )

            # Background Replace row
            rep_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            rep_lbl = Gtk.Label(label="Background Replace")
            rep_lbl.set_hexpand(True)
            rep_lbl.set_halign(Gtk.Align.START)
            rep_row.append(rep_lbl)
            self.bg_preset_dd = Gtk.DropDown.new_from_strings(list(BG_REPLACE_PRESETS))
            cur_preset = ceff["background_replace"].get("preset", "studio_dark")
            if cur_preset in BG_REPLACE_PRESETS:
                self.bg_preset_dd.set_selected(BG_REPLACE_PRESETS.index(cur_preset))
            self.bg_preset_dd.connect("notify::selected", self._on_bg_preset_changed)
            rep_row.append(self.bg_preset_dd)
            self.sw_replace = Gtk.Switch(active=bool(ceff["background_replace"]["enabled"]))
            self.sw_replace.set_valign(Gtk.Align.CENTER)
            self.sw_replace.connect("state-set", lambda _s, st: self._set_cam_bg_effect("background_replace", st, None))
            rep_row.append(self.sw_replace)
            fx_card.append(rep_row)

            # Background Remove row
            rem_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            rem_lbl = Gtk.Label(label="Background Removal (Chroma Green)")
            rem_lbl.set_hexpand(True)
            rem_lbl.set_halign(Gtk.Align.START)
            rem_row.append(rem_lbl)
            self.sw_remove = Gtk.Switch(active=bool(ceff["background_remove"]["enabled"]))
            self.sw_remove.set_valign(Gtk.Align.CENTER)
            self.sw_remove.connect("state-set", lambda _s, st: self._set_cam_bg_effect("background_remove", st, None))
            rem_row.append(self.sw_remove)
            fx_card.append(rem_row)

            self._add_toggle_slider_row(
                fx_card,
                "Auto Frame",
                "Smooth face-tracking crop & zoom",
                ceff["auto_frame"]["enabled"],
                ceff["auto_frame"]["zoom"],
                lambda en, val: self._set_generic_effect("auto_frame", en, {"zoom": val}),
            )
            self._add_toggle_slider_row(
                fx_card,
                "Video Noise Removal",
                "Reduces grainy low-light webcam sensor noise",
                ceff["video_noise_removal"]["enabled"],
                ceff["video_noise_removal"]["strength"],
                lambda en, val: self._set_generic_effect("video_noise_removal", en, {"strength": val}),
            )
            self._add_toggle_slider_row(
                fx_card,
                "Virtual Key Light",
                "Warmth: 0% Cool Daylight -> 100% Warm Studio",
                ceff["virtual_key_light"]["enabled"],
                ceff["virtual_key_light"]["warmth"],
                lambda en, val: self._set_generic_effect("virtual_key_light", en, {"warmth": val}),
            )
            self._add_toggle_slider_row(
                fx_card,
                "Vignette (Face-Tracked)",
                "Subtle studio edge darkening",
                ceff["vignette"]["enabled"],
                ceff["vignette"]["strength"],
                lambda en, val: self._set_generic_effect("vignette", en, {"strength": val}),
            )
            right.append(fx_card)

            # V4L2 Hardware Controls card
            hw_card = self._make_card()
            hw_lbl = Gtk.Label(label="Webcam Hardware Controls (V4L2)")
            hw_lbl.add_css_class("nv-title")
            hw_lbl.set_halign(Gtk.Align.START)
            hw_card.append(hw_lbl)

            cam_path = cams[0].path if cams else "/dev/video0"
            hw_controls = list_v4l2_controls(cam_path)
            wanted_slugs = {"brightness", "contrast", "saturation", "gain", "sharpness", "power_line_frequency"}
            shown = [c for c in hw_controls if c.slug in wanted_slugs]
            if shown:
                for ctrl in shown:
                    crow = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
                    cl = Gtk.Label(label=ctrl.name)
                    cl.add_css_class("nv-subtitle")
                    cl.set_size_request(145, -1)
                    cl.set_halign(Gtk.Align.START)
                    crow.append(cl)
                    sc = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, ctrl.minimum, ctrl.maximum, max(1, ctrl.step))
                    sc.set_value(ctrl.value)
                    sc.set_hexpand(True)
                    sc.connect("value-changed", self._make_v4l2_handler(cam_path, ctrl.id, ctrl.slug))
                    crow.append(sc)
                    hw_card.append(crow)
            else:
                empty_lbl = Gtk.Label(label="No adjustable V4L2 controls detected.")
                empty_lbl.add_css_class("nv-subtitle")
                hw_card.append(empty_lbl)
            right.append(hw_card)

            split.append(right_scroll)
            return split

        def _build_status_bar(self) -> Gtk.Widget:
            bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=18)
            bar.add_css_class("nv-status-bar")
            caps = probe_capabilities()
            gpu_txt = f"GPU: {caps.gpu.name}" if caps.gpu.available else "GPU: CPU Mode"
            pw_txt = f"PipeWire {caps.pipewire_version}" if caps.pipewire_version else "PipeWire N/A"
            cam_txt = "v4l2loopback: Ready" if caps.v4l2loopback_loaded else "v4l2loopback: Preview-Only"
            ai_txt = f"Audio: {caps.recommended_audio_backend} | Seg: {caps.recommended_segmentation_backend}"

            for txt, ok in ((gpu_txt, caps.gpu.available), (pw_txt, bool(caps.pipewire_version)), (cam_txt, caps.v4l2loopback_loaded), (ai_txt, True)):
                l = Gtk.Label(label=txt)
                l.add_css_class("nv-badge-ok" if ok else "nv-badge-warn")
                bar.append(l)
            return bar

        # ---- Helpers ----
        def _make_card(self) -> Gtk.Box:
            b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            b.add_css_class("nv-card")
            return b

        def _add_toggle_slider_row(
            self,
            parent: Gtk.Box,
            title: str,
            subtitle: str,
            enabled: bool,
            value: float,
            on_change: Any,
        ) -> tuple[Gtk.Switch, Gtk.Scale]:
            outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            tb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
            tb.set_hexpand(True)
            t = Gtk.Label(label=title)
            t.set_halign(Gtk.Align.START)
            s = Gtk.Label(label=subtitle)
            s.add_css_class("nv-subtitle")
            s.set_halign(Gtk.Align.START)
            tb.append(t)
            tb.append(s)
            top.append(tb)

            sw = Gtk.Switch(active=bool(enabled))
            sw.set_valign(Gtk.Align.CENTER)
            top.append(sw)
            outer.append(top)

            scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.05)
            scale.set_value(float(value))
            scale.set_draw_value(True)
            outer.append(scale)

            def _changed(*_args: Any) -> None:
                if self._updating_ui:
                    return
                on_change(sw.get_active(), round(scale.get_value(), 2))

            sw.connect("state-set", lambda _w, st: (on_change(st, round(scale.get_value(), 2)) if not self._updating_ui else None) or False)
            scale.connect("value-changed", _changed)
            parent.append(outer)
            return sw, scale

        # ---- Callbacks ----
        def _set_mic_flag(self, key: str, value: bool) -> None:
            cfg.data["microphone"][key] = value
            cfg.save()

        def _set_mic_effect(self, eid: str, enabled: bool, params: dict[str, Any] | None) -> None:
            cfg.set_effect(eid, enabled=enabled, params=params)
            cfg.save()
            self._sync_mic_switches()

        def _on_sv_toggled(self, _sw: Gtk.Switch, state: bool) -> bool:
            if not self._updating_ui:
                idx = self.sv_profile_dd.get_selected()
                prof = STUDIO_VOICE_PROFILES[idx] if idx < len(STUDIO_VOICE_PROFILES) else "default"
                cfg.set_effect("studio_voice", enabled=state, params={"micProfile": prof})
                cfg.save()
                self._sync_mic_switches()
            return False

        def _on_sv_changed(self, *_args: Any) -> None:
            if self._updating_ui:
                return
            idx = self.sv_profile_dd.get_selected()
            prof = STUDIO_VOICE_PROFILES[idx] if idx < len(STUDIO_VOICE_PROFILES) else "default"
            cfg.set_effect("studio_voice", params={"micProfile": prof})
            cfg.save()

        def _sync_mic_switches(self) -> None:
            self._updating_ui = True
            try:
                effs = cfg.data["microphone"]["effects"]
                self.sw_nr.set_active(bool(effs["noise_removal"]["enabled"]))
                self.sw_echo.set_active(bool(effs["room_echo_removal"]["enabled"]))
                self.sw_sv.set_active(bool(effs["studio_voice"]["enabled"]))
            finally:
                self._updating_ui = False

        def _set_generic_effect(self, eid: str, enabled: bool, params: dict[str, Any] | None) -> None:
            cfg.set_effect(eid, enabled=enabled, params=params)
            cfg.save()

        def _set_cam_bg_effect(self, eid: str, enabled: bool, params: dict[str, Any] | None) -> bool:
            if not self._updating_ui:
                cfg.set_effect(eid, enabled=enabled, params=params)
                cfg.save()
                self._updating_ui = True
                try:
                    ceff = cfg.data["camera"]["effects"]
                    self.sw_blur.set_active(bool(ceff["background_blur"]["enabled"]))
                    self.sw_replace.set_active(bool(ceff["background_replace"]["enabled"]))
                    self.sw_remove.set_active(bool(ceff["background_remove"]["enabled"]))
                finally:
                    self._updating_ui = False
            return False

        def _on_bg_preset_changed(self, *_args: Any) -> None:
            idx = self.bg_preset_dd.get_selected()
            preset = BG_REPLACE_PRESETS[idx] if idx < len(BG_REPLACE_PRESETS) else "studio_dark"
            cfg.set_effect("background_replace", params={"preset": preset})
            cfg.save()

        def _on_mic_device_changed(self, *_args: Any) -> None:
            idx = self.mic_dropdown.get_selected()
            if idx < len(self._mic_ids):
                cfg.data["microphone"]["device"] = self._mic_ids[idx]
                cfg.save()

        def _on_spk_device_changed(self, *_args: Any) -> None:
            idx = self.spk_dropdown.get_selected()
            if idx < len(self._spk_ids):
                cfg.data["speaker"]["device"] = self._spk_ids[idx]
                cfg.save()

        def _on_cam_device_changed(self, *_args: Any) -> None:
            idx = self.cam_dropdown.get_selected()
            if idx < len(self._cam_paths):
                cfg.data["camera"]["device"] = self._cam_paths[idx]
                cfg.save()

        def _make_v4l2_handler(self, dev: str, cid: int, slug: str) -> Any:
            def _cb(scale: Gtk.Scale) -> None:
                val = int(round(scale.get_value()))
                set_v4l2_control(dev, cid, val)
                cfg.data["camera"].setdefault("v4l2_controls", {})[slug] = val
                cfg.save()

            return _cb

        def _on_toggle_mic_engine(self, _btn: Gtk.Button) -> None:
            if audio_engine.mic_running:
                audio_engine.stop_microphone()
                self.mic_toggle_btn.set_label("Start Virtual Mic")
                self.mic_toggle_btn.remove_css_class("nv-stop-btn")
                self.mic_toggle_btn.add_css_class("nv-accent-btn")
            else:
                ok, _ = audio_engine.start_microphone()
                if ok:
                    self.mic_toggle_btn.set_label("Stop Virtual Mic")
                    self.mic_toggle_btn.remove_css_class("nv-accent-btn")
                    self.mic_toggle_btn.add_css_class("nv-stop-btn")

        def _on_toggle_spk_engine(self, _btn: Gtk.Button) -> None:
            if audio_engine.speaker_running:
                audio_engine.stop_speakers()
                self.spk_toggle_btn.set_label("Start Virtual Speakers")
                self.spk_toggle_btn.remove_css_class("nv-stop-btn")
                self.spk_toggle_btn.add_css_class("nv-accent-btn")
            else:
                ok, _ = audio_engine.start_speakers()
                if ok:
                    self.spk_toggle_btn.set_label("Stop Virtual Speakers")
                    self.spk_toggle_btn.remove_css_class("nv-accent-btn")
                    self.spk_toggle_btn.add_css_class("nv-stop-btn")

        def _on_toggle_cam_engine(self, _btn: Gtk.Button) -> None:
            if video_engine.running:
                video_engine.stop()
                self.cam_toggle_btn.set_label("Start Camera")
                self.cam_toggle_btn.remove_css_class("nv-stop-btn")
                self.cam_toggle_btn.add_css_class("nv-accent-btn")
                self.cam_fps_lbl.set_label("Idle")
            else:
                ok, msg = video_engine.start(preview_only=False)
                if ok:
                    self.cam_toggle_btn.set_label("Stop Camera")
                    self.cam_toggle_btn.remove_css_class("nv-accent-btn")
                    self.cam_toggle_btn.add_css_class("nv-stop-btn")
                    self.cam_fps_lbl.set_label("Starting...")
                else:
                    self.cam_fps_lbl.set_label(msg)

        def _on_record_mic_test(self, _btn: Gtk.Button) -> None:
            self.btn_rec_test.set_sensitive(False)
            self.test_status_lbl.set_label("Recording 5 seconds... speak normally with background noise.")

            def _worker() -> None:
                try:
                    res = audio_engine.record_mic_test(duration_sec=5.0)
                    self._last_mic_test = res
                    GLib.idle_add(self._on_mic_test_done, res, None)
                except Exception as exc:
                    GLib.idle_add(self._on_mic_test_done, None, str(exc))

            threading.Thread(target=_worker, daemon=True).start()

        def _on_mic_test_done(self, res: MicTestResult | None, err: str | None) -> bool:
            self.btn_rec_test.set_sensitive(True)
            if err or res is None:
                self.test_status_lbl.set_label(f"Mic test failed: {err}")
            else:
                self.btn_play_raw.set_sensitive(True)
                self.btn_play_proc.set_sensitive(True)
                self.test_status_lbl.set_label(
                    f"Recorded {res.duration_sec}s — Raw: {res.raw_rms_db} dBFS | Processed: {res.processed_rms_db} dBFS"
                )
            return False

        def _play_test(self, processed: bool) -> None:
            if not self._last_mic_test:
                return
            path = self._last_mic_test.processed_wav_path if processed else self._last_mic_test.raw_wav_path
            audio_engine.play_wav(path)

        def _on_audio_meter(self, meter: AudioMeterReading) -> None:
            frac = max(0.0, min(1.0, (meter.output_rms_db + 70.0) / 70.0))

            def _update() -> bool:
                self.vu_bar.set_fraction(frac)
                self.vu_db_lbl.set_label(f"{meter.output_rms_db:5.1f} dBFS")
                return False

            GLib.idle_add(_update)

        def _on_video_frame(self, pil_img: Any) -> None:
            small = pil_img.resize((640, 360))
            raw = small.tobytes()
            w, h = small.size

            def _update() -> bool:
                gbytes = GLib.Bytes.new(raw)
                texture = Gdk.MemoryTexture.new(w, h, Gdk.MemoryFormat.R8G8B8, gbytes, w * 3)
                self.preview_picture.set_paintable(texture)
                self.cam_fps_lbl.set_label(f"{video_engine.fps:.1f} FPS")
                return False

            GLib.idle_add(_update)

        def _on_close(self, *_args: Any) -> bool:
            audio_engine.stop_all()
            video_engine.stop()
            return False

    app = Adw.Application(application_id="io.github.broadcast.Linux")

    def _on_activate(a: Adw.Application) -> None:
        win = BroadcastWindow(a)
        win.present()

    app.connect("activate", _on_activate)
    return int(app.run([]))
