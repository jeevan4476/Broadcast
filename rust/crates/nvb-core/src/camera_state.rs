//! Explicit camera lifecycle state machine.
//!
//! Replaces overlapping boolean flags (`_running`, `_standby_active`,
//! `_on_demand_active`, `_manual_active`) with a single enum so mutually
//! impossible states cannot arise.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraState {
    /// Camera and standby frame loop are completely stopped.
    Off,
    /// Standby banner is displayed on v4l2loopback while `/proc` watches for apps.
    /// Zero physical camera access and zero GPU VRAM allocated.
    Standby,
    /// Automatically woken by one or more external v4l2loopback consumers.
    OnDemandActive { consumers: usize },
    /// Explicitly started by the user via UI/CLI; stays active regardless of consumers.
    ManualActive,
}

impl CameraState {
    pub fn is_capture_running(self) -> bool {
        matches!(self, Self::OnDemandActive { .. } | Self::ManualActive)
    }

    pub fn is_standby(self) -> bool {
        matches!(self, Self::Standby)
    }

    pub fn is_on_demand_active(self) -> bool {
        matches!(self, Self::OnDemandActive { .. })
    }

    pub fn is_manual_active(self) -> bool {
        matches!(self, Self::ManualActive)
    }

    pub fn enable_on_demand(self) -> Self {
        match self {
            Self::Off => Self::Standby,
            other => other,
        }
    }

    pub fn disable_on_demand(self) -> Self {
        match self {
            Self::Standby | Self::OnDemandActive { .. } => Self::Off,
            other => other,
        }
    }

    pub fn start_manual(self) -> Self {
        Self::ManualActive
    }

    pub fn stop_manual(self, on_demand_enabled: bool) -> Self {
        match self {
            Self::ManualActive => {
                if on_demand_enabled {
                    Self::Standby
                } else {
                    Self::Off
                }
            }
            other => other,
        }
    }

    /// Apply a consumer count observation from the `/proc` loopback monitor.
    pub fn on_consumers_observed(
        self,
        consumer_count: usize,
        debounce_expired: bool,
        on_demand_enabled: bool,
    ) -> Self {
        if !on_demand_enabled {
            return self;
        }
        match self {
            Self::ManualActive => Self::ManualActive,
            Self::Off | Self::Standby => {
                if consumer_count > 0 {
                    Self::OnDemandActive {
                        consumers: consumer_count,
                    }
                } else {
                    Self::Standby
                }
            }
            Self::OnDemandActive { .. } => {
                if consumer_count > 0 {
                    Self::OnDemandActive {
                        consumers: consumer_count,
                    }
                } else if debounce_expired {
                    Self::Standby
                } else {
                    Self::OnDemandActive { consumers: 0 }
                }
            }
        }
    }

    pub fn shutdown(self) -> Self {
        Self::Off
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_demand_wake_and_sleep_cycle() {
        let s = CameraState::Off.enable_on_demand();
        assert_eq!(s, CameraState::Standby);
        assert!(s.is_standby());
        assert!(!s.is_capture_running());

        let s = s.on_consumers_observed(1, false, true);
        assert_eq!(s, CameraState::OnDemandActive { consumers: 1 });
        assert!(s.is_capture_running());
        assert!(s.is_on_demand_active());

        // Consumers drop to 0, debounce not yet expired -> stay active
        let s = s.on_consumers_observed(0, false, true);
        assert_eq!(s, CameraState::OnDemandActive { consumers: 0 });
        assert!(s.is_capture_running());

        // Debounce expires -> return to Standby (0 MiB VRAM)
        let s = s.on_consumers_observed(0, true, true);
        assert_eq!(s, CameraState::Standby);
        assert!(!s.is_capture_running());
    }

    #[test]
    fn manual_mode_ignores_debounce_and_returns_to_standby_on_stop() {
        let s = CameraState::Standby.start_manual();
        assert_eq!(s, CameraState::ManualActive);
        assert!(s.is_manual_active());

        // Zero consumers + expired debounce must not stop manual capture
        let s = s.on_consumers_observed(0, true, true);
        assert_eq!(s, CameraState::ManualActive);

        // Stopping manual with on_demand=true restores Standby
        assert_eq!(s.stop_manual(true), CameraState::Standby);
        assert_eq!(s.stop_manual(false), CameraState::Off);
    }

    #[test]
    fn shutdown_always_reaches_off() {
        assert_eq!(CameraState::Standby.shutdown(), CameraState::Off);
        assert_eq!(CameraState::ManualActive.shutdown(), CameraState::Off);
        assert_eq!(
            CameraState::OnDemandActive { consumers: 2 }.shutdown(),
            CameraState::Off
        );
    }
}
