//! Phase R0 scaffold for the NVIDIA Broadcast (Linux) Rust daemon (`nvb-daemon`).
//! Phase R2 replaces this stub with the live `axum`/`tokio` HTTP + SSE + MJPEG server.

use nvb_core::api_contract::HealthResponse;
use nvb_core::{APP_NAME, APP_VERSION};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("{APP_NAME} nvb-daemon {APP_VERSION}");
        return;
    }
    let health = HealthResponse::ok();
    let json = serde_json::to_string(&health).expect("health serializes");
    println!("nvb-daemon scaffold ready: {json}");
}
