//! Phase R0 scaffold for the isolated on-demand GPU camera worker (`nvb-camera-worker`).
//! Spawned only while a consumer reads the virtual camera so idle daemon RAM/VRAM stay minimal.

use nvb_core::{APP_NAME, APP_VERSION};
use nvb_video::CameraState;

fn main() {
    let initial = CameraState::Standby;
    println!("{APP_NAME} nvb-camera-worker {APP_VERSION} (initial={initial:?})");
}
