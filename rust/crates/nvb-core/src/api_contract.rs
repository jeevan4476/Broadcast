//! HTTP + SSE JSON wire contracts matching `nv_broadcast/server.py`.

use crate::config::BroadcastConfig;
use crate::{APP_NAME, APP_VERSION};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const AUTH_HEADER: &str = "X-NVB-Token";
pub const AUTH_QUERY_PARAM: &str = "token";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub version: String,
}

impl HealthResponse {
    pub fn ok() -> Self {
        Self {
            ok: true,
            version: APP_VERSION.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
}

impl Default for AppInfo {
    fn default() -> Self {
        Self {
            name: APP_NAME.to_string(),
            version: APP_VERSION.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioMeterReading {
    pub input_rms_db: f64,
    pub output_rms_db: f64,
    pub input_peak_db: f64,
    pub output_peak_db: f64,
    pub noise_reduction_db: f64,
    pub speech_prob: f64,
    pub gate_open: bool,
    pub backend: String,
}

impl Default for AudioMeterReading {
    fn default() -> Self {
        Self {
            input_rms_db: -90.0,
            output_rms_db: -90.0,
            input_peak_db: -90.0,
            output_peak_db: -90.0,
            noise_reduction_db: 0.0,
            speech_prob: 0.0,
            gate_open: false,
            backend: "dsp".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioStreamTelemetry {
    pub running: bool,
    pub backend: String,
    pub is_default: bool,
    pub meter: AudioMeterReading,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraConsumer {
    pub pid: u32,
    pub comm: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraTelemetry {
    pub running: bool,
    pub fps: f64,
    pub input: String,
    pub output: String,
    pub resolution: String,
    pub segmentation_backend: String,
    pub face_backend: String,
    pub last_error: String,
    pub on_demand: bool,
    pub idle_timeout_sec: f64,
    pub on_demand_active: bool,
    pub external_consumers: Vec<CameraConsumer>,
    pub standby_active: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub mic: AudioStreamTelemetry,
    pub speaker: AudioStreamTelemetry,
    pub camera: CameraTelemetry,
    pub gpu: Value,
    pub device_sig: String,
    pub file_job: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceDefaults {
    pub default_source: String,
    pub default_sink: String,
    pub mic_is_default: bool,
    pub speaker_is_default: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevicesPayload {
    pub microphones: Vec<Value>,
    pub speakers: Vec<Value>,
    pub cameras: Vec<Value>,
    pub loopback_cameras: Vec<Value>,
    pub bluetooth_cards: Vec<Value>,
    pub defaults: DeviceDefaults,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullStatePayload {
    pub app: AppInfo,
    pub config: BroadcastConfig,
    pub catalog: Value,
    pub incompatible_groups: Vec<Vec<String>>,
    pub devices: DevicesPayload,
    pub virtual_nodes: Value,
    pub capabilities: Value,
    pub models: Value,
    pub telemetry: TelemetrySnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostEffectResponse {
    pub ok: bool,
    pub applied: Vec<String>,
    pub disabled_conflicts: BTreeMap<String, Vec<String>>,
    pub errors: Vec<String>,
    pub config: BroadcastConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::INCOMPATIBLE_GROUPS;

    const API_ORACLE_JSON: &str = include_str!("../../../fixtures/api_contract_oracle.json");

    fn sorted_keys(val: &Value) -> Vec<String> {
        let mut keys: Vec<String> = val.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    }

    fn string_vec(val: &Value) -> Vec<String> {
        val.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn api_contract_matches_python_oracle() {
        let oracle: Value = serde_json::from_str(API_ORACLE_JSON).unwrap();

        assert_eq!(AUTH_HEADER, oracle["auth_header"].as_str().unwrap());
        assert_eq!(
            AUTH_QUERY_PARAM,
            oracle["auth_query_param"].as_str().unwrap()
        );

        let err = serde_json::to_value(ErrorResponse {
            error: "unauthorized".to_string(),
        })
        .unwrap();
        assert_eq!(err, oracle["unauthorized_response"]["body"]);
        assert_eq!(sorted_keys(&err), string_vec(&oracle["unauthorized_response"]["keys"]));

        let health = serde_json::to_value(HealthResponse::ok()).unwrap();
        assert_eq!(health, oracle["health"]["body"]);
        assert_eq!(sorted_keys(&health), string_vec(&oracle["health"]["keys"]));

        let devices = DevicesPayload {
            microphones: Vec::new(),
            speakers: Vec::new(),
            cameras: Vec::new(),
            loopback_cameras: Vec::new(),
            bluetooth_cards: Vec::new(),
            defaults: DeviceDefaults {
                default_source: String::new(),
                default_sink: String::new(),
                mic_is_default: false,
                speaker_is_default: false,
            },
        };
        let devices_val = serde_json::to_value(&devices).unwrap();
        assert_eq!(
            sorted_keys(&devices_val),
            string_vec(&oracle["devices"]["top_level_keys"])
        );
        assert_eq!(
            sorted_keys(&devices_val["defaults"]),
            string_vec(&oracle["devices"]["defaults_keys"])
        );

        let state = FullStatePayload {
            app: AppInfo::default(),
            config: BroadcastConfig::default(),
            catalog: Value::Object(Default::default()),
            incompatible_groups: INCOMPATIBLE_GROUPS
                .iter()
                .map(|g| g.iter().map(|s| s.to_string()).collect())
                .collect(),
            devices,
            virtual_nodes: Value::Object(Default::default()),
            capabilities: Value::Object(Default::default()),
            models: Value::Object(Default::default()),
            telemetry: TelemetrySnapshot {
                mic: AudioStreamTelemetry {
                    running: false,
                    backend: "dsp".to_string(),
                    is_default: false,
                    meter: AudioMeterReading::default(),
                },
                speaker: AudioStreamTelemetry {
                    running: false,
                    backend: "dsp".to_string(),
                    is_default: false,
                    meter: AudioMeterReading::default(),
                },
                camera: CameraTelemetry {
                    running: false,
                    fps: 0.0,
                    input: String::new(),
                    output: String::new(),
                    resolution: String::new(),
                    segmentation_backend: "standby".to_string(),
                    face_backend: "standby".to_string(),
                    last_error: String::new(),
                    on_demand: true,
                    idle_timeout_sec: 3.0,
                    on_demand_active: false,
                    external_consumers: Vec::new(),
                    standby_active: false,
                },
                gpu: Value::Object(Default::default()),
                device_sig: String::new(),
                file_job: Value::Object(Default::default()),
            },
        };
        let state_val = serde_json::to_value(&state).unwrap();
        assert_eq!(
            sorted_keys(&state_val),
            string_vec(&oracle["state"]["top_level_keys"])
        );
        assert_eq!(
            sorted_keys(&state_val["app"]),
            string_vec(&oracle["state"]["app_keys"])
        );
        assert_eq!(
            sorted_keys(&state_val["telemetry"]),
            string_vec(&oracle["state"]["telemetry_keys"])
        );
        assert_eq!(
            sorted_keys(&state_val["telemetry"]["mic"]),
            string_vec(&oracle["state"]["telemetry_mic_keys"])
        );
        assert_eq!(
            sorted_keys(&state_val["telemetry"]["speaker"]),
            string_vec(&oracle["state"]["telemetry_speaker_keys"])
        );
        assert_eq!(
            sorted_keys(&state_val["telemetry"]["camera"]),
            string_vec(&oracle["state"]["telemetry_camera_keys"])
        );
        assert_eq!(
            sorted_keys(&state_val["telemetry"]["mic"]["meter"]),
            string_vec(&oracle["state"]["meter_keys"])
        );

        let mut cfg = BroadcastConfig::default();
        let mut params = serde_json::Map::new();
        params.insert("micProfile".to_string(), Value::String("warm".to_string()));
        let res = cfg.set_effect("studio_voice", Some(true), Some(&params), true);
        let post_resp = PostEffectResponse {
            ok: res.errors.is_empty(),
            applied: res.applied,
            disabled_conflicts: res.disabled_conflicts,
            errors: res.errors,
            config: cfg,
        };
        let post_val = serde_json::to_value(&post_resp).unwrap();
        assert_eq!(
            sorted_keys(&post_val),
            string_vec(&oracle["post_effect"]["keys"])
        );
        assert_eq!(post_val["ok"], oracle["post_effect"]["ok"]);
        assert_eq!(post_val["applied"], oracle["post_effect"]["applied"]);
        assert_eq!(
            post_val["disabled_conflicts"],
            oracle["post_effect"]["disabled_conflicts"]
        );
        assert_eq!(post_val["errors"], oracle["post_effect"]["errors"]);
    }
}
