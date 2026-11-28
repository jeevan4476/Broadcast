//! Pure-Rust Linux `/proc` scanner for v4l2loopback external consumers.
//!
//! Port of `nv_broadcast/loopback_monitor.py` with zero subprocess spawning.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopbackConsumer {
    pub pid: u32,
    pub comm: String,
}

fn read_proc_comm(pid_dir: &Path) -> String {
    fs::read_to_string(pid_dir.join("comm"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Return external processes holding open file descriptors pointing to `device_path`.
///
/// Scans `proc_root/<pid>/fd/*` directly via `std::fs` (zero subprocess overhead).
pub fn list_loopback_consumers(
    device_path: &str,
    exclude_pids: Option<&HashSet<u32>>,
    proc_root: &Path,
) -> Vec<LoopbackConsumer> {
    if device_path.is_empty() {
        return Vec::new();
    }

    let target: PathBuf = fs::canonicalize(device_path).unwrap_or_else(|_| PathBuf::from(device_path));
    let raw_target = Path::new(device_path);
    let self_pid = process::id();

    let Ok(entries) = fs::read_dir(proc_root) else {
        return Vec::new();
    };

    let mut consumers = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Ok(pid) = name_str.parse::<u32>() else {
            continue;
        };
        let excluded = match exclude_pids {
            Some(set) => set.contains(&pid),
            None => pid == self_pid,
        };
        if excluded {
            continue;
        }

        let pid_dir = entry.path();
        let fd_dir = pid_dir.join("fd");
        let Ok(fds) = fs::read_dir(&fd_dir) else {
            continue;
        };

        let mut matched = false;
        for fd_entry in fds.flatten() {
            if let Ok(link) = fs::read_link(fd_entry.path()) {
                if link == target || link == raw_target {
                    matched = true;
                    break;
                }
            }
        }

        if matched {
            consumers.push(LoopbackConsumer {
                pid,
                comm: read_proc_comm(&pid_dir),
            });
        }
    }

    consumers.sort_by_key(|c| c.pid);
    consumers
}

/// Convenience helper checking whether at least one external process has `device_path` open.
pub fn has_external_consumers(
    device_path: &str,
    exclude_pids: Option<&HashSet<u32>>,
    proc_root: &Path,
) -> bool {
    !list_loopback_consumers(device_path, exclude_pids, proc_root).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn scans_synthetic_proc_and_excludes_own_pid() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("nvb_proc_test_{nonce}"));
        let p100 = root.join("100");
        let p200 = root.join("200");
        let p300 = root.join("300");
        fs::create_dir_all(p100.join("fd")).unwrap();
        fs::create_dir_all(p200.join("fd")).unwrap();
        fs::create_dir_all(p300.join("fd")).unwrap();

        fs::write(p100.join("comm"), "ffmpeg\n").unwrap();
        fs::write(p200.join("comm"), "discord\n").unwrap();
        fs::write(p300.join("comm"), "chrome\n").unwrap();

        symlink("/dev/video10", p100.join("fd/3")).unwrap();
        symlink("/dev/video10", p200.join("fd/7")).unwrap();
        symlink("/dev/video0", p300.join("fd/4")).unwrap();

        let mut exclude = HashSet::new();
        exclude.insert(100);

        let found = list_loopback_consumers("/dev/video10", Some(&exclude), &root);
        assert_eq!(
            found,
            vec![LoopbackConsumer {
                pid: 200,
                comm: "discord".to_string()
            }]
        );
        assert!(has_external_consumers("/dev/video10", Some(&exclude), &root));
        assert!(!has_external_consumers("/dev/video99", Some(&exclude), &root));

        let _ = fs::remove_dir_all(&root);
    }
}
