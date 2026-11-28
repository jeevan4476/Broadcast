//! Core configuration, typed invariants, state machines, device probing,
//! and wire contracts for NVIDIA Broadcast (Linux).

pub mod api_contract;
pub mod camera_state;
pub mod config;
pub mod devices;
pub mod doctor;
pub mod loopback_monitor;
pub mod models;
pub mod routing;

pub const APP_NAME: &str = "NVIDIA Broadcast (Linux)";
pub const APP_VERSION: &str = "2.3.0-linux.1";
