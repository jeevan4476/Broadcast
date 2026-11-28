//! Audio DSP utilities, pure-Rust neural audio processor, and PipeWire anti-loop routing contracts.

pub mod dsp;
pub mod engine;
pub mod file_processor;

use std::collections::HashMap;

pub use dsp::{
    build_ffmpeg_audio_filter, compute_waveform_peaks, generate_sample_clip, read_wav_mono16,
    write_wav_mono16, AudioProcessConfig, AudioProcessor,
};
pub use engine::{AudioEngine, PipeWireVirtualManager};
pub use file_processor::{
    has_enabled_video_effects, FileJobState, FileProcessor, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS,
};

pub use nvb_core::api_contract::AudioMeterReading;
pub use nvb_core::routing::{
    is_virtual_broadcast_node, AudioRoute, PhysicalDevice, RoutingError, VirtualDevice,
    VIRTUAL_MIC_SINK_NAME, VIRTUAL_MIC_SOURCE_NAME, VIRTUAL_NODE_NAMES, VIRTUAL_SPEAKER_SINK_NAME,
};

pub const SAMPLE_RATE: u32 = 48_000;
pub const RNNOISE_FRAME_SAMPLES: usize = 480;
pub const DEFAULT_LATENCY_MS: u32 = 20;

/// Convert little-endian signed 16-bit PCM bytes into normalized float samples in [-1.0, 1.0).
pub fn pcm16_to_f32(pcm_bytes: &[u8]) -> Vec<f32> {
    let inv = 1.0_f64 / 32768.0_f64;
    pcm_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&chunk| {
            let sample = i16::from_le_bytes(chunk);
            (f64::from(sample) * inv) as f32
        })
        .collect()
}

/// Convert normalized float samples into little-endian signed 16-bit PCM bytes,
/// matching Python's `int(round(s * 32767.0))` (round-half-to-even).
pub fn f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let scaled = (f64::from(s) * 32767.0).round_ties_even();
        let clamped = scaled.clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&clamped.to_le_bytes());
    }
    out
}

/// Compute `(rms_dbfs, peak_dbfs)` for normalized float samples in `[-1.0, 1.0]`.
pub fn compute_rms_db(samples: &[f32]) -> (f64, f64) {
    if samples.is_empty() {
        return (-90.0, -90.0);
    }
    let mut sq = 0.0_f64;
    let mut peak = 0.0_f64;
    for &s in samples {
        let sf = f64::from(s);
        let a = sf.abs();
        sq += sf * sf;
        if a > peak {
            peak = a;
        }
    }
    let rms = (sq / samples.len() as f64).sqrt();
    let rms_db = 20.0 * rms.max(1e-45).log10();
    let peak_db = 20.0 * peak.max(1e-45).log10();
    (rms_db.max(-90.0), peak_db.max(-90.0))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PwCatMode {
    Record,
    Playback,
}

/// Build a `pw-cat` argument vector isolated from WirePlumber auto-routing/reconnection.
///
/// Setting both `node.autoconnect = false` and `node.dont-reconnect = true` prevents
/// WirePlumber from moving a Broadcast stream onto another sink's monitor or back
/// into a virtual sink (preventing acoustic and digital feedback loops).
pub fn build_isolated_pw_cat_args(
    mode: PwCatMode,
    node_name: &str,
    rate: u32,
    channels: u32,
    latency_ms: u32,
) -> Vec<String> {
    let flag = match mode {
        PwCatMode::Record => "--record",
        PwCatMode::Playback => "--playback",
    };
    let props = format!(
        "{{ node.name = \"{node_name}\" node.autoconnect = false node.dont-reconnect = true }}"
    );
    vec![
        "pw-cat".to_string(),
        flag.to_string(),
        "--raw".to_string(),
        "--format=s16".to_string(),
        format!("--rate={rate}"),
        format!("--channels={channels}"),
        format!("--latency={latency_ms}ms"),
        "-P".to_string(),
        props,
        "-".to_string(),
    ]
}

/// Parse `pw-link -l` stdout into directed `(src_port, dst_port)` pairs.
pub fn parse_pw_links(output: &str) -> Vec<(String, String)> {
    let mut edges = Vec::new();
    let mut current_src = String::new();
    for line in output.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let first_is_ws = line.chars().next().is_some_and(char::is_whitespace);
        if !first_is_ws {
            current_src = line.trim().to_string();
        } else if let Some((_, rhs)) = line.split_once("|->") {
            let dst = rhs.trim();
            if !current_src.is_empty() && !dst.is_empty() {
                edges.push((current_src.clone(), dst.to_string()));
            }
        }
    }
    edges
}

/// Return true if at least one active link connects `src_node` -> `dst_node`.
pub fn is_pw_link_active(src_node: &str, dst_node: &str, links: &[(String, String)]) -> bool {
    if src_node.is_empty() || dst_node.is_empty() {
        return false;
    }
    let src_prefix = format!("{src_node}:");
    let dst_prefix = format!("{dst_node}:");
    links
        .iter()
        .any(|(s, d)| s.starts_with(&src_prefix) && d.starts_with(&dst_prefix))
}

fn port_channel(port: &str) -> String {
    let suffix = port.split_once(':').map_or(port, |(_, s)| s);
    match suffix.rsplit_once('_') {
        Some((_, ch)) => ch.to_ascii_uppercase(),
        None => suffix.to_ascii_uppercase(),
    }
}

/// Deterministically pair PipeWire ports between `src_node` and `dst_node`.
pub fn select_port_pairs(
    src_node: &str,
    dst_node: &str,
    out_ports: &[String],
    in_ports: &[String],
    src_prefix: &str,
    dst_prefix: &str,
) -> Vec<(String, String)> {
    let src_head = format!("{src_node}:");
    let dst_head = format!("{dst_node}:");

    let src_list: Vec<&str> = out_ports
        .iter()
        .filter(|p| p.starts_with(&src_head) && p[src_head.len()..].starts_with(src_prefix))
        .map(String::as_str)
        .collect();
    let dst_list: Vec<&str> = in_ports
        .iter()
        .filter(|p| p.starts_with(&dst_head) && p[dst_head.len()..].starts_with(dst_prefix))
        .map(String::as_str)
        .collect();

    if src_list.is_empty() || dst_list.is_empty() {
        return Vec::new();
    }

    let mut src_by_ch: HashMap<String, &str> = HashMap::new();
    for &p in &src_list {
        src_by_ch.insert(port_channel(p), p);
    }
    let mut dst_by_ch: HashMap<String, &str> = HashMap::new();
    for &p in &dst_list {
        dst_by_ch.insert(port_channel(p), p);
    }

    let mut pairs = Vec::new();
    for ch in ["FL", "FR", "MONO"] {
        if let (Some(&sp), Some(&dp)) = (src_by_ch.get(ch), dst_by_ch.get(ch)) {
            pairs.push((sp.to_string(), dp.to_string()));
        }
    }
    if !pairs.is_empty() {
        return pairs;
    }

    // Stereo source -> single mono input (use FL / first channel to avoid +6 dB summing)
    if dst_list.len() == 1 {
        let sp = src_by_ch.get("FL").copied().unwrap_or(src_list[0]);
        return vec![(sp.to_string(), dst_list[0].to_string())];
    }

    // Mono source -> stereo sink (feed both L and R)
    if src_list.len() == 1 {
        return dst_list
            .iter()
            .take(2)
            .map(|&dp| (src_list[0].to_string(), dp.to_string()))
            .collect();
    }

    // Positional fallback (e.g., AUX0/AUX1)
    src_list
        .into_iter()
        .zip(dst_list)
        .map(|(s, d)| (s.to_string(), d.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::f64::consts::PI;

    const DSP_ORACLE_JSON: &str = include_str!("../../../fixtures/dsp_oracle.json");

    fn oracle() -> Value {
        serde_json::from_str(DSP_ORACLE_JSON).expect("valid dsp_oracle.json")
    }

    fn hex_encode(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            use std::fmt::Write;
            let _ = write!(&mut out, "{b:02x}");
        }
        out
    }

    fn pairs_from_json(val: &Value) -> Vec<(String, String)> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let arr = pair.as_array().unwrap();
                (
                    arr[0].as_str().unwrap().to_string(),
                    arr[1].as_str().unwrap().to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn rms_and_pcm16_match_python_oracle() {
        let o = oracle();
        assert_eq!(
            o["audio_constants"]["sample_rate"].as_u64().unwrap(),
            u64::from(SAMPLE_RATE)
        );
        assert_eq!(
            o["audio_constants"]["rnnoise_frame_samples"]
                .as_u64()
                .unwrap(),
            RNNOISE_FRAME_SAMPLES as u64
        );
        assert_eq!(
            serde_json::to_value(AudioMeterReading::default()).unwrap(),
            o["audio_constants"]["default_meter"]
        );

        for case in o["rms_cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let samples: Vec<f32> = if case["is_sine_440hz_480"].as_bool().unwrap() {
                (0..480)
                    .map(|i| (0.5 * (2.0 * PI * 440.0 * f64::from(i) / 48000.0).sin()) as f32)
                    .collect()
            } else {
                case["samples"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            };

            let (rms_db, peak_db) = compute_rms_db(&samples);
            let exp_rms = case["rms_db"].as_f64().unwrap();
            let exp_peak = case["peak_db"].as_f64().unwrap();
            assert!(
                (rms_db - exp_rms).abs() < 1e-4,
                "{name}: rms mismatch got {rms_db} expected {exp_rms}"
            );
            assert!(
                (peak_db - exp_peak).abs() < 1e-4,
                "{name}: peak mismatch got {peak_db} expected {exp_peak}"
            );

            let pcm = f32_to_pcm16(&samples);
            assert_eq!(
                pcm.len(),
                case["pcm16_len_bytes"].as_u64().unwrap() as usize,
                "{name}: pcm16 length mismatch"
            );
            assert_eq!(
                hex_encode(&pcm[..pcm.len().min(16)]),
                case["pcm16_head_hex"].as_str().unwrap(),
                "{name}: pcm16 hex head mismatch"
            );

            let roundtrip = pcm16_to_f32(&pcm);
            let exp_rt: Vec<f64> = case["roundtrip_head"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            for (idx, exp) in exp_rt.iter().enumerate() {
                assert!(
                    (f64::from(roundtrip[idx]) - exp).abs() < 1e-6,
                    "{name}[{idx}]: roundtrip mismatch"
                );
            }
        }
    }

    #[test]
    fn pipewire_helpers_match_python_oracle() {
        let o = oracle();
        let pw = &o["pipewire"];

        assert_eq!(
            pw["virtual_mic_sink_name"].as_str().unwrap(),
            VIRTUAL_MIC_SINK_NAME
        );
        assert_eq!(
            pw["virtual_mic_source_name"].as_str().unwrap(),
            VIRTUAL_MIC_SOURCE_NAME
        );
        assert_eq!(
            pw["virtual_speaker_sink_name"].as_str().unwrap(),
            VIRTUAL_SPEAKER_SINK_NAME
        );

        for (node, expected) in pw["virtual_checks"].as_object().unwrap() {
            assert_eq!(
                is_virtual_broadcast_node(node),
                expected.as_bool().unwrap(),
                "virtual check mismatch for '{node}'"
            );
        }

        let rec_cmd = build_isolated_pw_cat_args(
            PwCatMode::Record,
            "nv_broadcast_mic_capture",
            SAMPLE_RATE,
            1,
            DEFAULT_LATENCY_MS,
        );
        assert_eq!(
            serde_json::to_value(&rec_cmd).unwrap(),
            pw["isolated_pw_cat_record"]
        );

        let play_cmd = build_isolated_pw_cat_args(
            PwCatMode::Playback,
            "nv_broadcast_spk_playback",
            SAMPLE_RATE,
            2,
            DEFAULT_LATENCY_MS,
        );
        assert_eq!(
            serde_json::to_value(&play_cmd).unwrap(),
            pw["isolated_pw_cat_playback"]
        );

        let parsed = parse_pw_links(pw["sample_pw_links_raw"].as_str().unwrap());
        assert_eq!(parsed, pairs_from_json(&pw["parsed_pw_links"]));
        assert_eq!(
            is_pw_link_active(
                "alsa_input.pci-0000_00_1f.3",
                "nv_broadcast_mic_capture",
                &parsed
            ),
            pw["link_active_true"].as_bool().unwrap()
        );
        assert_eq!(
            is_pw_link_active("alsa_input.pci-0000_00_1f.3", "other_app", &parsed),
            pw["link_active_false"].as_bool().unwrap()
        );

        let pp = &pw["port_pairing"];
        let out_ports: Vec<String> = pp["out_ports"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let in_ports: Vec<String> = pp["in_ports"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        assert_eq!(
            select_port_pairs("phys_mic", "cap_stereo", &out_ports, &in_ports, "", ""),
            pairs_from_json(&pp["stereo_to_stereo"])
        );
        assert_eq!(
            select_port_pairs("phys_mic", "cap_mono", &out_ports, &in_ports, "", ""),
            pairs_from_json(&pp["stereo_to_mono"])
        );
        assert_eq!(
            select_port_pairs("mono_mic", "cap_stereo", &out_ports, &in_ports, "", ""),
            pairs_from_json(&pp["mono_to_stereo"])
        );
        assert_eq!(
            select_port_pairs("aux_src", "cap_aux", &out_ports, &in_ports, "", ""),
            pairs_from_json(&pp["aux_positional"])
        );
    }
}
