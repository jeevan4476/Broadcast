//! Core configuration, typed invariants, state machines, and wire contracts
//! for NVIDIA Broadcast (Linux).

pub mod api_contract;
pub mod camera_state;
pub mod config;
pub mod routing;

pub const APP_NAME: &str = "NVIDIA Broadcast (Linux)";
pub const APP_VERSION: &str = "2.3.0-linux.1";
