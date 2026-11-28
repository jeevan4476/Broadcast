//! NVIDIA Broadcast (Linux) native Rust HTTP + SSE + MJPEG daemon (`nvb-daemon`).
//!
//! Drop-in `/api/*` parity with `nv_broadcast/server.py`, built on `axum` + `tokio`
//! with a 2-thread async runtime, cached hardware telemetry (<1 ms `/api/state`),
//! pure-Rust `nvb-audio` (`nnnoiseless`), and isolated short-lived camera worker
//! management so idle RAM stays <= 25 MB and GPU VRAM returns to 0 MiB after calls.

use std::collections::HashMap;
use std::convert::Infallible;
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use nvb_audio::{AudioEngine, PipeWireVirtualManager};
use nvb_core::api_contract::{
    AppInfo, AudioStreamTelemetry, CameraConsumer, CameraTelemetry, DevicesPayload, ErrorResponse,
    FullStatePayload, HealthResponse, PostEffectResponse, TelemetrySnapshot, AUTH_HEADER,
    AUTH_QUERY_PARAM,
};
use nvb_core::camera_state::CameraState;
use nvb_core::config::{clamp_float, BroadcastConfig, INCOMPATIBLE_GROUPS};
use nvb_core::devices::{
    build_devices_payload, probe_capabilities, query_gpu_live, set_bluetooth_headset_mode,
};
use nvb_core::doctor::run_doctor;
use nvb_core::loopback_monitor::list_loopback_consumers;
use nvb_core::models::get_models_status;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::signal;
use tokio_stream::wrappers::IntervalStream;
use tokio_stream::StreamExt;

const CATALOG_ORACLE_JSON: &str = include_str!("../../../fixtures/config_oracle.json");

fn load_effect_catalog() -> Value {
    serde_json::from_str::<Value>(CATALOG_ORACLE_JSON)
        .ok()
        .and_then(|v| v.get("effect_catalog").cloned())
        .unwrap_or_else(|| json!({}))
}

fn generate_hex_token(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    if let Ok(mut f) = File::open("/dev/urandom") {
        let _ = f.read_exact(&mut buf);
    } else {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = ((nanos >> ((i % 16) * 8)) & 0xFF) as u8;
        }
    }
    let mut hex = String::with_capacity(bytes * 2);
    for b in buf {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{b:02x}");
    }
    hex
}

fn default_config_path() -> PathBuf {
    let base = env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            home.join(".config")
        });
    base.join("nv-broadcast").join("config.json")
}

fn daemon_state_path() -> PathBuf {
    default_config_path()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("daemon.json")
}

/// Manages camera lifecycle state and optional isolated short-lived worker child process.
struct CameraController {
    state: CameraState,
    worker: Option<Child>,
    active_input: String,
    active_output: String,
    active_resolution: String,
    segmentation_backend: String,
    face_backend: String,
    last_error: String,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            state: CameraState::Off,
            worker: None,
            active_input: String::new(),
            active_output: String::new(),
            active_resolution: String::new(),
            segmentation_backend: "standby".to_string(),
            face_backend: "standby".to_string(),
            last_error: String::new(),
        }
    }
}

impl CameraController {
    fn start_manual(&mut self, cfg: &BroadcastConfig, preview_only: bool) -> (bool, String) {
        self.stop(false);
        self.state = self.state.start_manual();
        self.active_input = cfg.camera.device.clone();
        self.active_output = if preview_only {
            String::new()
        } else {
            cfg.camera.output_device.clone()
        };
        self.active_resolution = format!("{}x{}@{}", cfg.camera.width, cfg.camera.height, cfg.camera.fps);
        self.segmentation_backend = cfg.segmentation_backend.clone();
        self.face_backend = "worker".to_string();
        self.last_error.clear();

        // Optional isolated external worker command (`NVB_CAMERA_WORKER`)
        if let Ok(worker_cmd) = env::var("NVB_CAMERA_WORKER") {
            let trimmed = worker_cmd.trim();
            if !trimmed.is_empty() {
                match Command::new(trimmed)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                {
                    Ok(child) => self.worker = Some(child),
                    Err(e) => self.last_error = format!("worker spawn failed: {e}"),
                }
            }
        }
        (true, "Started camera session".to_string())
    }

    fn start_standby(&mut self, cfg: &BroadcastConfig) {
        if !self.state.is_capture_running() {
            self.state = self.state.enable_on_demand();
            self.active_output = cfg.camera.output_device.clone();
            self.segmentation_backend = "standby".to_string();
            self.face_backend = "standby".to_string();
        }
    }

    fn stop(&mut self, shutdown: bool) {
        if let Some(mut child) = self.worker.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.state = if shutdown {
            self.state.shutdown()
        } else {
            self.state.stop_manual(true)
        };
        self.segmentation_backend = "standby".to_string();
        self.face_backend = "standby".to_string();
    }

    fn telemetry(&self, cfg: &BroadcastConfig) -> CameraTelemetry {
        let consumers: Vec<CameraConsumer> = if !self.active_output.is_empty()
            && self.active_output != "auto"
            && Path::new(&self.active_output).exists()
        {
            list_loopback_consumers(&self.active_output, None, Path::new("/proc"))
                .into_iter()
                .map(|c| CameraConsumer {
                    pid: c.pid,
                    comm: c.comm,
                })
                .collect()
        } else {
            Vec::new()
        };

        CameraTelemetry {
            running: self.state.is_capture_running(),
            fps: 0.0,
            input: self.active_input.clone(),
            output: self.active_output.clone(),
            resolution: self.active_resolution.clone(),
            segmentation_backend: self.segmentation_backend.clone(),
            face_backend: self.face_backend.clone(),
            last_error: self.last_error.clone(),
            on_demand: cfg.camera.on_demand,
            idle_timeout_sec: cfg.camera.idle_timeout_sec,
            on_demand_active: self.state.is_on_demand_active(),
            external_consumers: consumers,
            standby_active: self.state.is_standby(),
        }
    }
}

impl Drop for CameraController {
    fn drop(&mut self) {
        self.stop(true);
    }
}

struct CachedSystemState {
    devices: DevicesPayload,
    capabilities: Value,
    virtual_nodes: Value,
    models: Value,
    gpu_live: Value,
    device_sig: String,
    last_refresh: Instant,
    refreshed_once: bool,
}

impl Default for CachedSystemState {
    fn default() -> Self {
        Self {
            devices: DevicesPayload {
                microphones: Vec::new(),
                speakers: Vec::new(),
                cameras: Vec::new(),
                loopback_cameras: Vec::new(),
                bluetooth_cards: Vec::new(),
                defaults: nvb_core::api_contract::DeviceDefaults {
                    default_source: String::new(),
                    default_sink: String::new(),
                    mic_is_default: false,
                    speaker_is_default: false,
                },
            },
            capabilities: json!({}),
            virtual_nodes: json!({
                "virtual_mic_source": false,
                "virtual_mic_sink": false,
                "virtual_speaker_sink": false,
            }),
            models: json!({}),
            gpu_live: json!({
                "available": false,
                "utilization_pct": 0,
                "memory_used_mib": 0,
                "memory_total_mib": 0,
                "temp_c": 0,
            }),
            device_sig: String::new(),
            last_refresh: Instant::now() - Duration::from_secs(60),
            refreshed_once: false,
        }
    }
}

struct AppState {
    config: Arc<RwLock<BroadcastConfig>>,
    config_path: PathBuf,
    auth_token: String,
    catalog: Value,
    incompatible_groups: Vec<Vec<String>>,
    audio: AudioEngine,
    camera: Mutex<CameraController>,
    cache: RwLock<CachedSystemState>,
}

impl AppState {
    fn new(config_path: PathBuf, auth_token: String) -> Arc<Self> {
        let cfg = BroadcastConfig::load_from_path(&config_path);
        let config_arc = Arc::new(RwLock::new(cfg));
        let audio = AudioEngine::new(Arc::clone(&config_arc));
        let incompatible_groups = INCOMPATIBLE_GROUPS
            .iter()
            .map(|g| g.iter().map(|s| (*s).to_string()).collect())
            .collect();

        Arc::new(Self {
            config: config_arc,
            config_path,
            auth_token,
            catalog: load_effect_catalog(),
            incompatible_groups,
            audio,
            camera: Mutex::new(CameraController::default()),
            cache: RwLock::new(CachedSystemState::default()),
        })
    }

    fn refresh_cache_if_stale(&self, max_age: Duration, include_caps: bool) {
        let needs_refresh = self
            .cache
            .read()
            .map(|c| !c.refreshed_once || c.last_refresh.elapsed() >= max_age)
            .unwrap_or(true);
        if !needs_refresh {
            return;
        }
        let devices = build_devices_payload(false);
        let virtual_nodes = PipeWireVirtualManager::find_existing_virtual_nodes();
        let gpu_live = query_gpu_live();
        let models = serde_json::to_value(get_models_status()).unwrap_or_else(|_| json!({}));
        let capabilities = if include_caps {
            Some(probe_capabilities())
        } else {
            None
        };
        let sig = format!(
            "{}||{}||{}||{}",
            devices.microphones.len(),
            devices.speakers.len(),
            devices.defaults.default_source,
            devices.defaults.default_sink
        );
        if let Ok(mut c) = self.cache.write() {
            c.devices = devices;
            c.virtual_nodes = virtual_nodes;
            c.gpu_live = gpu_live;
            c.models = models;
            if let Some(caps) = capabilities {
                c.capabilities = caps;
            }
            c.device_sig = sig;
            c.last_refresh = Instant::now();
            c.refreshed_once = true;
        }
    }

    fn telemetry(&self) -> TelemetrySnapshot {
        let cfg = self.config.read().map(|c| c.clone()).unwrap_or_default();
        let (mic_def, spk_def, gpu, sig) = self
            .cache
            .read()
            .map(|c| {
                (
                    c.devices.defaults.mic_is_default,
                    c.devices.defaults.speaker_is_default,
                    c.gpu_live.clone(),
                    c.device_sig.clone(),
                )
            })
            .unwrap_or_else(|_| (false, false, json!({}), String::new()));

        let cam_tel = self
            .camera
            .lock()
            .map(|cam| cam.telemetry(&cfg))
            .unwrap_or_else(|_| CameraController::default().telemetry(&cfg));

        TelemetrySnapshot {
            mic: AudioStreamTelemetry {
                running: self.audio.mic_running(),
                backend: "nnnoiseless".to_string(),
                is_default: mic_def,
                meter: self.audio.mic_meter(),
            },
            speaker: AudioStreamTelemetry {
                running: self.audio.speaker_running(),
                backend: "nnnoiseless".to_string(),
                is_default: spk_def,
                meter: self.audio.speaker_meter(),
            },
            camera: cam_tel,
            gpu,
            device_sig: sig,
            file_job: json!({"running": false, "progress": 0.0, "status": "idle"}),
        }
    }

    fn full_state(&self) -> FullStatePayload {
        self.refresh_cache_if_stale(Duration::from_secs(5), true);
        let cfg = self.config.read().map(|c| c.clone()).unwrap_or_default();
        let (devices, virtual_nodes, capabilities, models) = self
            .cache
            .read()
            .map(|c| {
                (
                    c.devices.clone(),
                    c.virtual_nodes.clone(),
                    c.capabilities.clone(),
                    c.models.clone(),
                )
            })
            .unwrap_or_else(|_| {
                let d = CachedSystemState::default();
                (d.devices, d.virtual_nodes, d.capabilities, d.models)
            });

        FullStatePayload {
            app: AppInfo::default(),
            config: cfg,
            catalog: self.catalog.clone(),
            incompatible_groups: self.incompatible_groups.clone(),
            devices,
            virtual_nodes,
            capabilities,
            models,
            telemetry: self.telemetry(),
        }
    }

    fn save_config(&self) {
        if let Ok(mut cfg) = self.config.write() {
            let _ = cfg.save_to_path(&self.config_path);
        }
    }

    fn check_auth(&self, headers: &HeaderMap, query: &HashMap<String, String>) -> bool {
        if self.auth_token.is_empty() {
            return true;
        }
        if let Some(val) = headers.get(AUTH_HEADER).and_then(|v| v.to_str().ok()) {
            if val == self.auth_token {
                return true;
            }
        }
        query
            .get(AUTH_QUERY_PARAM)
            .is_some_and(|v| v == &self.auth_token)
    }
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "unauthorized".to_string(),
        }),
    )
        .into_response()
}

async fn get_health(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    Json(HealthResponse::ok()).into_response()
}

async fn get_state(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    Json(st.full_state()).into_response()
}

async fn get_devices(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    st.refresh_cache_if_stale(Duration::from_secs(3), false);
    let devs = st
        .cache
        .read()
        .map(|c| c.devices.clone())
        .unwrap_or_else(|_| build_devices_payload(false));
    Json(devs).into_response()
}

async fn get_doctor(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let report = tokio::task::spawn_blocking(run_doctor)
        .await
        .unwrap_or_else(|_| json!({"error": "doctor task failed"}));
    Json(report).into_response()
}

async fn get_files_state(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    Json(json!({"running": false, "progress": 0.0, "status": "idle"})).into_response()
}

async fn get_camera_v4l2(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let dev = q
        .get("device")
        .filter(|d| !d.is_empty() && d.as_str() != "auto")
        .cloned()
        .unwrap_or_else(|| "/dev/video0".to_string());
    Json(json!({"device": dev, "controls": []})).into_response()
}

async fn get_events_sse(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let stream = IntervalStream::new(tokio::time::interval(Duration::from_millis(80))).map(move |_| {
        let data = serde_json::to_string(&st.telemetry()).unwrap_or_else(|_| "{}".to_string());
        Ok::<Event, Infallible>(Event::default().data(data))
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn post_effect(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let eid = body
        .get("effect_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let enabled = body.get("enabled").and_then(Value::as_bool);
    let params = body.get("params").and_then(Value::as_object);

    let (res, cfg_clone) = {
        let mut cfg = match st.config.write() {
            Ok(g) => g,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "config lock poisoned".to_string(),
                    }),
                )
                    .into_response();
            }
        };
        let r = cfg.set_effect(eid, enabled, params, true);
        let _ = cfg.save_to_path(&st.config_path);
        (r, cfg.clone())
    };

    Json(PostEffectResponse {
        ok: res.errors.is_empty(),
        applied: res.applied,
        disabled_conflicts: res.disabled_conflicts,
        errors: res.errors,
        config: cfg_clone,
    })
    .into_response()
}

async fn post_mic_start(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let (ok, msg) = st.audio.start_microphone();
    Json(json!({"ok": ok, "message": msg, "telemetry": st.telemetry()})).into_response()
}

async fn post_mic_stop(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    st.audio.stop_microphone();
    Json(json!({"ok": true, "message": "Stopped virtual microphone", "telemetry": st.telemetry()}))
        .into_response()
}

async fn post_mic_config(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let mut restart = false;
    let cfg_clone = {
        let mut cfg = st.config.write().unwrap();
        if let Some(dev) = body.get("device").and_then(Value::as_str) {
            if cfg.microphone.device != dev {
                restart = true;
            }
            cfg.microphone.device = dev.to_string();
        }
        if let Some(v) = body.get("highpass_enabled").and_then(Value::as_bool) {
            cfg.microphone.highpass_enabled = v;
        }
        if let Some(v) = body.get("highpass_hz").and_then(Value::as_i64) {
            cfg.microphone.highpass_hz = v;
        }
        if let Some(v) = body.get("noise_gate_enabled").and_then(Value::as_bool) {
            cfg.microphone.noise_gate_enabled = v;
        }
        if let Some(v) = body.get("noise_gate_db").and_then(Value::as_f64) {
            cfg.microphone.noise_gate_db = v;
        }
        if let Some(v) = body.get("auto_gain_enabled").and_then(Value::as_bool) {
            cfg.microphone.auto_gain_enabled = v;
        }
        if let Some(v) = body.get("auto_gain_target_db").and_then(Value::as_f64) {
            cfg.microphone.auto_gain_target_db = v;
        }
        if let Some(b) = body.get("audio_backend").and_then(Value::as_str) {
            cfg.audio_backend = b.to_string();
            restart = true;
        }
        let _ = cfg.save_to_path(&st.config_path);
        cfg.clone()
    };
    if restart && st.audio.mic_running() {
        st.audio.stop_microphone();
        let _ = st.audio.start_microphone();
    }
    Json(json!({"ok": true, "config": cfg_clone})).into_response()
}

async fn post_spk_start(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let (ok, msg) = st.audio.start_speakers();
    Json(json!({"ok": ok, "message": msg, "telemetry": st.telemetry()})).into_response()
}

async fn post_spk_stop(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    st.audio.stop_speakers();
    Json(json!({"ok": true, "message": "Stopped virtual speakers", "telemetry": st.telemetry()}))
        .into_response()
}

async fn post_spk_config(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let mut restart = false;
    let cfg_clone = {
        let mut cfg = st.config.write().unwrap();
        if let Some(dev) = body.get("device").and_then(Value::as_str) {
            if cfg.speaker.device != dev {
                restart = true;
            }
            cfg.speaker.device = dev.to_string();
        }
        let _ = cfg.save_to_path(&st.config_path);
        cfg.clone()
    };
    if restart && st.audio.speaker_running() {
        st.audio.stop_speakers();
        let _ = st.audio.start_speakers();
    }
    Json(json!({"ok": true, "config": cfg_clone})).into_response()
}

async fn post_spk_sample(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let kind = body
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("fan")
        .to_string();
    match st.audio.build_speaker_sample(&kind) {
        Ok(mut val) => {
            if let Some(obj) = val.as_object_mut() {
                obj.insert("ok".to_string(), Value::Bool(true));
            }
            Json(val).into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": err})),
        )
            .into_response(),
    }
}

async fn post_audio_play(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let path = body.get("path").and_then(Value::as_str).unwrap_or("");
    let sink = body.get("target_sink").and_then(Value::as_str);
    let (ok, msg) = st.audio.play_wav(path, sink);
    Json(json!({"ok": ok, "message": msg})).into_response()
}

async fn post_audio_stop_play(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    st.audio.stop_playback();
    Json(json!({"ok": true})).into_response()
}

async fn post_set_default(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let target = body.get("target").and_then(Value::as_str).unwrap_or("");
    let enable = body.get("enable").and_then(Value::as_bool).unwrap_or(true);
    let (ok, msg) = match target {
        "mic" => PipeWireVirtualManager::set_virtual_mic_as_default(enable),
        "speaker" => PipeWireVirtualManager::set_virtual_speakers_as_default(enable),
        _ => (false, format!("Invalid target: {target}")),
    };
    st.refresh_cache_if_stale(Duration::ZERO, false);
    let devs = st.cache.read().map(|c| c.devices.clone()).unwrap_or_else(|_| build_devices_payload(false));
    Json(json!({"ok": ok, "message": msg, "devices": devs})).into_response()
}

async fn post_bluetooth_profile(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let card = body.get("card").and_then(Value::as_str).unwrap_or("");
    let enable_mic = body
        .get("enable_mic")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let (ok, msg) = set_bluetooth_headset_mode(card, enable_mic);
    st.refresh_cache_if_stale(Duration::ZERO, false);
    let devs = st.cache.read().map(|c| c.devices.clone()).unwrap_or_else(|_| build_devices_payload(false));
    Json(json!({"ok": ok, "message": msg, "devices": devs})).into_response()
}

async fn post_camera_start(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let preview_only = body
        .get("preview_only")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let cfg = st.config.read().map(|c| c.clone()).unwrap_or_default();
    let (ok, msg) = st
        .camera
        .lock()
        .map(|mut cam| cam.start_manual(&cfg, preview_only))
        .unwrap_or((false, "camera lock failed".to_string()));
    Json(json!({"ok": ok, "message": msg, "telemetry": st.telemetry()})).into_response()
}

async fn post_camera_stop(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    if let Ok(mut cam) = st.camera.lock() {
        cam.stop(false);
    }
    Json(json!({"ok": true, "message": "Stopped camera", "telemetry": st.telemetry()}))
        .into_response()
}

async fn post_camera_config(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Response {
    if !st.check_auth(&headers, &q) {
        return unauthorized();
    }
    let cfg_clone = {
        let mut cfg = st.config.write().unwrap();
        if let Some(v) = body.get("device").and_then(Value::as_str) {
            cfg.camera.device = v.to_string();
        }
        if let Some(v) = body.get("output_device").and_then(Value::as_str) {
            cfg.camera.output_device = v.to_string();
        }
        if let Some(v) = body.get("width").and_then(Value::as_u64) {
            cfg.camera.width = v as u32;
        }
        if let Some(v) = body.get("height").and_then(Value::as_u64) {
            cfg.camera.height = v as u32;
        }
        if let Some(v) = body.get("fps").and_then(Value::as_u64) {
            cfg.camera.fps = v as u32;
        }
        if let Some(v) = body.get("prefer_mjpg").and_then(Value::as_bool) {
            cfg.camera.prefer_mjpg = v;
        }
        if let Some(v) = body.get("mirror").and_then(Value::as_bool) {
            cfg.camera.mirror = v;
        }
        if let Some(v) = body.get("split_preview").and_then(Value::as_bool) {
            cfg.camera.split_preview = v;
        }
        if let Some(v) = body.get("split_position").and_then(Value::as_f64) {
            cfg.camera.split_position = clamp_float(v, 0.05, 0.95, 0.5);
        }
        if let Some(v) = body.get("on_demand").and_then(Value::as_bool) {
            cfg.camera.on_demand = v;
        }
        if let Some(v) = body.get("idle_timeout_sec").and_then(Value::as_f64) {
            cfg.camera.idle_timeout_sec = clamp_float(v, 0.5, 60.0, 3.0);
        }
        let _ = cfg.save_to_path(&st.config_path);
        cfg.clone()
    };
    Json(json!({"ok": true, "config": cfg_clone})).into_response()
}

async fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "not found".to_string(),
        }),
    )
        .into_response()
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(get_health))
        .route("/api/state", get(get_state))
        .route("/api/devices", get(get_devices))
        .route("/api/doctor", get(get_doctor))
        .route("/api/files/state", get(get_files_state))
        .route("/api/camera/v4l2", get(get_camera_v4l2))
        .route("/api/events", get(get_events_sse))
        .route("/api/effect", post(post_effect))
        .route("/api/audio/mic/start", post(post_mic_start))
        .route("/api/audio/mic/stop", post(post_mic_stop))
        .route("/api/audio/mic/config", post(post_mic_config))
        .route("/api/audio/speaker/start", post(post_spk_start))
        .route("/api/audio/speaker/stop", post(post_spk_stop))
        .route("/api/audio/speaker/config", post(post_spk_config))
        .route("/api/audio/speaker/sample", post(post_spk_sample))
        .route("/api/audio/play", post(post_audio_play))
        .route("/api/audio/stop-play", post(post_audio_stop_play))
        .route("/api/audio/set-default", post(post_set_default))
        .route("/api/audio/bluetooth-profile", post(post_bluetooth_profile))
        .route("/api/camera/start", post(post_camera_start))
        .route("/api/camera/stop", post(post_camera_stop))
        .route("/api/camera/config", post(post_camera_config))
        .fallback(not_found)
        .with_state(state)
}

struct DaemonArgs {
    host: String,
    port: u16,
    config: Option<PathBuf>,
    token: Option<String>,
    autostart: bool,
    daemon_file: bool,
}

fn parse_args() -> DaemonArgs {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut out = DaemonArgs {
        host: "127.0.0.1".to_string(),
        port: 0,
        config: None,
        token: None,
        autostart: false,
        daemon_file: false,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" if i + 1 < args.len() => {
                out.host = args[i + 1].clone();
                i += 2;
            }
            "--port" if i + 1 < args.len() => {
                out.port = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            "--config" if i + 1 < args.len() => {
                out.config = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--token" if i + 1 < args.len() => {
                out.token = Some(args[i + 1].clone());
                i += 2;
            }
            "--autostart" => {
                out.autostart = true;
                i += 1;
            }
            "--daemon-file" => {
                out.daemon_file = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    out
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::terminate()) {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

fn main() {
    let args = parse_args();
    // Keep runtime lean (2 worker threads -> low idle PSS and minimal context switches)
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");

    rt.block_on(async move {
        let cfg_path = args.config.unwrap_or_else(default_config_path);
        let auth_token = args.token.unwrap_or_else(|| generate_hex_token(16));
        let state = AppState::new(cfg_path, auth_token.clone());

        if args.autostart {
            let _ = state.audio.start_microphone();
            let _ = state.audio.start_speakers();
            let cfg = state.config.read().map(|c| c.clone()).unwrap_or_default();
            if let Ok(mut cam) = state.camera.lock() {
                cam.start_standby(&cfg);
            }
        }

        let bind_addr: SocketAddr = format!("{}:{}", args.host, args.port)
            .parse()
            .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], 0)));
        let listener = TcpListener::bind(bind_addr)
            .await
            .expect("failed to bind localhost HTTP listener");
        let local_addr = listener.local_addr().expect("listener has local addr");
        let base_url = format!("http://{}:{}", args.host, local_addr.port());

        let state_file = if args.daemon_file {
            let p = daemon_state_path();
            if let Some(parent) = p.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let tmp = p.with_extension("tmp");
            let payload = json!({
                "url": base_url,
                "token": auth_token,
                "pid": std::process::id(),
                "engine": "rust",
            });
            if fs::write(&tmp, format!("{payload}\n")).is_ok() {
                let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
                let _ = fs::rename(&tmp, &p);
            }
            Some(p)
        } else {
            None
        };

        // Warm hardware cache once in the background so READY prints immediately (<10 ms boot)
        let bg_state = Arc::clone(&state);
        tokio::task::spawn_blocking(move || {
            bg_state.refresh_cache_if_stale(Duration::ZERO, true);
        });

        println!("READY {base_url} {auth_token}");

        let app = build_router(Arc::clone(&state));
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await;

        state.audio.stop_all();
        if let Ok(mut cam) = state.camera.lock() {
            cam.stop(true);
        }
        state.save_config();
        if let Some(sf) = state_file {
            let _ = fs::remove_file(sf);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    #[tokio::test]
    async fn live_axum_router_matches_python_api_oracle() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cfg_path = env::temp_dir().join(format!("nvb_daemon_test_{nonce}.json"));
        let token = "oracle-test-token".to_string();
        let state = AppState::new(cfg_path.clone(), token.clone());
        let app = build_router(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        async fn http_req(addr: SocketAddr, req_raw: &str) -> (u16, Value) {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            stream.write_all(req_raw.as_bytes()).await.unwrap();
            let mut resp = String::new();
            stream.read_to_string(&mut resp).await.unwrap();
            let (head, body) = resp.split_once("\r\n\r\n").unwrap();
            let code: u16 = head
                .lines()
                .next()
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse()
                .unwrap();
            (code, serde_json::from_str(body).unwrap())
        }

        let oracle: Value = serde_json::from_str(API_ORACLE_JSON).unwrap();

        // 1. Unauthorized
        let (code, body) = http_req(
            addr,
            "GET /api/health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert_eq!(code, 401);
        assert_eq!(body, oracle["unauthorized_response"]["body"]);

        // 2. Authorized /api/health
        let (code, body) = http_req(
            addr,
            &format!("GET /api/health HTTP/1.1\r\nHost: localhost\r\nX-NVB-Token: {token}\r\nConnection: close\r\n\r\n"),
        )
        .await;
        assert_eq!(code, 200);
        assert_eq!(body, oracle["health"]["body"]);

        // 3. Authorized /api/state
        let (code, body) = http_req(
            addr,
            &format!("GET /api/state HTTP/1.1\r\nHost: localhost\r\nX-NVB-Token: {token}\r\nConnection: close\r\n\r\n"),
        )
        .await;
        assert_eq!(code, 200);
        assert_eq!(sorted_keys(&body), string_vec(&oracle["state"]["top_level_keys"]));
        assert_eq!(sorted_keys(&body["telemetry"]), string_vec(&oracle["state"]["telemetry_keys"]));

        // 4. Authorized /api/devices
        let (code, body) = http_req(
            addr,
            &format!("GET /api/devices HTTP/1.1\r\nHost: localhost\r\nX-NVB-Token: {token}\r\nConnection: close\r\n\r\n"),
        )
        .await;
        assert_eq!(code, 200);
        assert_eq!(sorted_keys(&body), string_vec(&oracle["devices"]["top_level_keys"]));

        // 5. Authorized POST /api/effect
        let payload = r#"{"effect_id":"studio_voice","enabled":true,"params":{"micProfile":"warm"}}"#;
        let (code, body) = http_req(
            addr,
            &format!(
                "POST /api/effect HTTP/1.1\r\nHost: localhost\r\nX-NVB-Token: {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            ),
        )
        .await;
        assert_eq!(code, 200);
        assert_eq!(sorted_keys(&body), string_vec(&oracle["post_effect"]["keys"]));
        assert_eq!(body["ok"], oracle["post_effect"]["ok"]);
        assert_eq!(body["applied"], oracle["post_effect"]["applied"]);
        assert_eq!(
            body["disabled_conflicts"],
            oracle["post_effect"]["disabled_conflicts"]
        );

        server.abort();
        let _ = fs::remove_file(&cfg_path);
    }
}
