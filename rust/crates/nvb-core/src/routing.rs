//! Typed PipeWire device routing invariants.
//!
//! Makes virtual-to-virtual PipeWire feedback loops unrepresentable in Rust's
//! type system by splitting physical devices from Broadcast's own virtual nodes.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const VIRTUAL_MIC_SINK_NAME: &str = "nv_broadcast_mic_bus";
pub const VIRTUAL_MIC_SOURCE_NAME: &str = "nv_broadcast_mic";
pub const VIRTUAL_SPEAKER_SINK_NAME: &str = "nv_broadcast_speakers";

pub const VIRTUAL_NODE_NAMES: &[&str] = &[
    VIRTUAL_MIC_SINK_NAME,
    VIRTUAL_MIC_SOURCE_NAME,
    VIRTUAL_SPEAKER_SINK_NAME,
];

/// Return true if a PipeWire node or device name belongs to Broadcast's own virtual nodes.
pub fn is_virtual_broadcast_node(name: &str) -> bool {
    let low = name.trim().to_ascii_lowercase();
    if low.is_empty() {
        return false;
    }
    VIRTUAL_NODE_NAMES.iter().any(|v| low.contains(v))
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RoutingError {
    #[error("device name must not be empty")]
    EmptyName,
    #[error("refusing to use Broadcast virtual node '{0}' as a physical device")]
    VirtualNodeRejected(String),
    #[error("device '{0}' is not a Broadcast virtual node")]
    NotVirtualNode(String),
}

/// A verified non-virtual physical PipeWire source or sink.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhysicalDevice(String);

impl PhysicalDevice {
    pub fn try_new(name: impl Into<String>) -> Result<Self, RoutingError> {
        let name = name.into();
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(RoutingError::EmptyName);
        }
        if is_virtual_broadcast_node(trimmed) {
            return Err(RoutingError::VirtualNodeRejected(trimmed.to_string()));
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A verified Broadcast-owned virtual PipeWire node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VirtualDevice(String);

impl VirtualDevice {
    pub fn try_new(name: impl Into<String>) -> Result<Self, RoutingError> {
        let name = name.into();
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(RoutingError::EmptyName);
        }
        if !is_virtual_broadcast_node(trimmed) {
            return Err(RoutingError::NotVirtualNode(trimmed.to_string()));
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Directed audio bridge between a physical device and a virtual Broadcast node.
/// Because the two endpoints have distinct newtypes, a virtual-to-virtual
/// feedback loop cannot be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioRoute {
    MicCapture {
        from: PhysicalDevice,
        to: VirtualDevice,
    },
    SpeakerPlayback {
        from: VirtualDevice,
        to: PhysicalDevice,
    },
}

impl AudioRoute {
    pub fn mic_capture(from: PhysicalDevice, to: VirtualDevice) -> Self {
        Self::MicCapture { from, to }
    }

    pub fn speaker_playback(from: VirtualDevice, to: PhysicalDevice) -> Self {
        Self::SpeakerPlayback { from, to }
    }

    pub fn src_node(&self) -> &str {
        match self {
            Self::MicCapture { from, .. } => from.as_str(),
            Self::SpeakerPlayback { from, .. } => from.as_str(),
        }
    }

    pub fn dst_node(&self) -> &str {
        match self {
            Self::MicCapture { to, .. } => to.as_str(),
            Self::SpeakerPlayback { to, .. } => to.as_str(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_device_rejects_virtual_nodes_and_empty() {
        assert_eq!(PhysicalDevice::try_new(""), Err(RoutingError::EmptyName));
        assert!(matches!(
            PhysicalDevice::try_new("nv_broadcast_mic"),
            Err(RoutingError::VirtualNodeRejected(_))
        ));
        assert!(matches!(
            PhysicalDevice::try_new("nv_broadcast_mic_bus.monitor"),
            Err(RoutingError::VirtualNodeRejected(_))
        ));
        assert!(matches!(
            PhysicalDevice::try_new("NV_BROADCAST_SPEAKERS"),
            Err(RoutingError::VirtualNodeRejected(_))
        ));

        let phys = PhysicalDevice::try_new("alsa_input.pci-0000_00_1f.3").unwrap();
        assert_eq!(phys.as_str(), "alsa_input.pci-0000_00_1f.3");
    }

    #[test]
    fn virtual_device_accepts_only_broadcast_nodes() {
        assert!(matches!(
            VirtualDevice::try_new("alsa_output.pci-0000_00_1f.3"),
            Err(RoutingError::NotVirtualNode(_))
        ));
        let virt = VirtualDevice::try_new(VIRTUAL_MIC_SINK_NAME).unwrap();
        assert_eq!(virt.as_str(), VIRTUAL_MIC_SINK_NAME);
    }

    #[test]
    fn audio_route_enforces_physical_virtual_separation() {
        let mic = PhysicalDevice::try_new("alsa_input.usb-mic").unwrap();
        let mic_bus = VirtualDevice::try_new(VIRTUAL_MIC_SINK_NAME).unwrap();
        let route = AudioRoute::mic_capture(mic, mic_bus);
        assert_eq!(route.src_node(), "alsa_input.usb-mic");
        assert_eq!(route.dst_node(), VIRTUAL_MIC_SINK_NAME);
    }
}
