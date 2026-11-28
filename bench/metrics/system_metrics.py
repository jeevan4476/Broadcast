"""Unprivileged Linux process-tree, GPU, PipeWire, and power telemetry metrics.

Reads directly from Linux `/proc` (`smaps_rollup`, `stat`, `status`, `fd`,
`children`), `nvidia-smi`, `pw-top`, `pw-link`, `/sys/class/power_supply/BAT*`,
and gracefully checks Intel RAPL without ever requiring root privileges.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def parse_smaps_rollup(text: str) -> dict[str, int]:
    """Parse `/proc/<pid>/smaps_rollup` into KiB integers."""
    wanted = {
        "Rss": "rss_kb",
        "Pss": "pss_kb",
        "Private_Clean": "private_clean_kb",
        "Private_Dirty": "private_dirty_kb",
        "Shared_Clean": "shared_clean_kb",
        "Shared_Dirty": "shared_dirty_kb",
        "SwapPss": "swap_pss_kb",
    }
    out = {v: 0 for v in wanted.values()}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        if key in wanted:
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                out[wanted[key]] = int(parts[0])
    return out


def parse_proc_status(text: str) -> dict[str, int]:
    """Parse `/proc/<pid>/status` for Threads, VmRSS, and context switch counters."""
    out = {
        "threads": 0,
        "vm_rss_kb": 0,
        "voluntary_ctxt_switches": 0,
        "nonvoluntary_ctxt_switches": 0,
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        parts = rest.strip().split()
        if not parts or not parts[0].isdigit():
            continue
        val = int(parts[0])
        if key == "Threads":
            out["threads"] = val
        elif key == "VmRSS":
            out["vm_rss_kb"] = val
        elif key == "voluntary_ctxt_switches":
            out["voluntary_ctxt_switches"] = val
        elif key == "nonvoluntary_ctxt_switches":
            out["nonvoluntary_ctxt_switches"] = val
    return out


def parse_proc_stat_cpu_ticks(text: str) -> tuple[int, int]:
    """Parse `(utime, stime)` clock ticks from `/proc/<pid>/stat`."""
    rparen = text.rfind(")")
    if rparen == -1:
        return 0, 0
    fields = text[rparen + 2 :].split()
    # After `(comm)`: index 0 is state (field 3), utime is field 14 -> idx 11, stime is field 15 -> idx 12
    if len(fields) < 13:
        return 0, 0
    try:
        return int(fields[11]), int(fields[12])
    except ValueError:
        return 0, 0


def get_process_tree_pids(root_pid: int) -> list[int]:
    """Recursively collect `root_pid` and all living descendant PIDs via `/proc`."""
    if root_pid <= 0 or not Path(f"/proc/{root_pid}").exists():
        return []

    children_map: dict[int, list[int]] = {}
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return [root_pid]

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_txt = (entry / "stat").read_text(encoding="utf-8", errors="ignore")
            rparen = stat_txt.rfind(")")
            if rparen == -1:
                continue
            fields = stat_txt[rparen + 2 :].split()
            if len(fields) >= 2:
                ppid = int(fields[1])
                children_map.setdefault(ppid, []).append(pid)
        except (OSError, ValueError):
            continue

    result: list[int] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        result.append(cur)
        stack.extend(children_map.get(cur, []))
    return result


def sample_process_tree(root_pid: int) -> dict[str, Any]:
    """Sample full memory, CPU ticks, threads, context switches, and open FDs across a PID tree."""
    pids = get_process_tree_pids(root_pid)
    totals: dict[str, Any] = {
        "root_pid": root_pid,
        "process_count": len(pids),
        "pids": pids,
        "rss_kb": 0,
        "pss_kb": 0,
        "private_dirty_kb": 0,
        "swap_pss_kb": 0,
        "threads": 0,
        "open_fds": 0,
        "voluntary_ctxt_switches": 0,
        "nonvoluntary_ctxt_switches": 0,
        "cpu_ticks": 0,
        "timestamp": time.monotonic(),
    }
    for pid in pids:
        pdir = Path(f"/proc/{pid}")
        try:
            status_txt = (pdir / "status").read_text(encoding="utf-8", errors="ignore")
            st = parse_proc_status(status_txt)
            totals["threads"] += st["threads"]
            totals["voluntary_ctxt_switches"] += st["voluntary_ctxt_switches"]
            totals["nonvoluntary_ctxt_switches"] += st["nonvoluntary_ctxt_switches"]
            fallback_rss = st["vm_rss_kb"]
        except OSError:
            fallback_rss = 0

        try:
            smaps_txt = (pdir / "smaps_rollup").read_text(encoding="utf-8", errors="ignore")
            sm = parse_smaps_rollup(smaps_txt)
            totals["rss_kb"] += sm["rss_kb"] or fallback_rss
            totals["pss_kb"] += sm["pss_kb"] or fallback_rss
            totals["private_dirty_kb"] += sm["private_dirty_kb"]
            totals["swap_pss_kb"] += sm["swap_pss_kb"]
        except OSError:
            totals["rss_kb"] += fallback_rss
            totals["pss_kb"] += fallback_rss

        try:
            stat_txt = (pdir / "stat").read_text(encoding="utf-8", errors="ignore")
            ut, kt = parse_proc_stat_cpu_ticks(stat_txt)
            totals["cpu_ticks"] += ut + kt
        except OSError:
            pass

        try:
            totals["open_fds"] += sum(1 for _ in (pdir / "fd").iterdir())
        except OSError:
            pass

    totals["rss_mb"] = round(totals["rss_kb"] / 1024.0, 2)
    totals["pss_mb"] = round(totals["pss_kb"] / 1024.0, 2)
    totals["private_dirty_mb"] = round(totals["private_dirty_kb"] / 1024.0, 2)
    return totals


def compute_process_delta(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    """Compute CPU-ms/s, % of 1 core, and context switches/s between two process-tree snapshots."""
    dt = max(1e-6, float(end["timestamp"]) - float(start["timestamp"]))
    d_ticks = max(0, int(end["cpu_ticks"]) - int(start["cpu_ticks"]))
    cpu_sec = d_ticks / float(CLK_TCK)
    cpu_ms_per_sec = (cpu_sec * 1000.0) / dt
    cpu_core_pct = (cpu_sec / dt) * 100.0
    d_vol = max(0, int(end["voluntary_ctxt_switches"]) - int(start["voluntary_ctxt_switches"]))
    d_nonvol = max(0, int(end["nonvoluntary_ctxt_switches"]) - int(start["nonvoluntary_ctxt_switches"]))
    return {
        "duration_sec": round(dt, 3),
        "cpu_ms_per_sec": round(cpu_ms_per_sec, 3),
        "cpu_core_pct": round(cpu_core_pct, 3),
        "voluntary_ctxt_per_sec": round(d_vol / dt, 2),
        "nonvoluntary_ctxt_per_sec": round(d_nonvol / dt, 2),
        "total_ctxt_per_sec": round((d_vol + d_nonvol) / dt, 2),
    }


def query_nvidia_gpu(filter_pids: list[int] | None = None) -> dict[str, Any]:
    """Query GPU name, P-state, power draw (W), total VRAM, and optional per-PID VRAM."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,pstate,power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return {"available": False, "reason": res.stderr.strip() or "nvidia-smi failed"}
        parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
        info: dict[str, Any] = {
            "available": True,
            "name": parts[0] if len(parts) > 0 else "unknown",
            "pstate": parts[1] if len(parts) > 1 else "unknown",
            "power_draw_w": _safe_float(parts[2]) if len(parts) > 2 else None,
            "gpu_vram_used_mib": _safe_float(parts[3]) if len(parts) > 3 else None,
            "gpu_util_pct": _safe_float(parts[4]) if len(parts) > 4 else None,
            "process_vram_mib": 0.0,
        }

        if filter_pids:
            pid_set = set(filter_pids)
            pres = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if pres.returncode == 0:
                vram = 0.0
                for line in pres.stdout.splitlines():
                    cols = [c.strip() for c in line.split(",")]
                    if len(cols) >= 2 and cols[0].isdigit() and int(cols[0]) in pid_set:
                        vram += _safe_float(cols[1]) or 0.0
                info["process_vram_mib"] = round(vram, 2)
        return info
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)}


def query_power_supply() -> dict[str, Any]:
    """Read Linux battery discharge draw (W) and AC status, plus Intel RAPL accessibility."""
    ps_dir = Path("/sys/class/power_supply")
    battery: dict[str, Any] = {"available": False}
    if ps_dir.is_dir():
        for bat in sorted(ps_dir.glob("BAT*")):
            status_p = bat / "status"
            power_p = bat / "power_now"
            curr_p = bat / "current_now"
            volt_p = bat / "voltage_now"
            try:
                status = status_p.read_text(encoding="utf-8").strip() if status_p.is_file() else "unknown"
                power_w: float | None = None
                if power_p.is_file():
                    power_w = round(int(power_p.read_text(encoding="utf-8").strip()) / 1_000_000.0, 3)
                elif curr_p.is_file() and volt_p.is_file():
                    ua = int(curr_p.read_text(encoding="utf-8").strip())
                    uv = int(volt_p.read_text(encoding="utf-8").strip())
                    power_w = round((ua * uv) / 1e12, 3)
                battery = {
                    "available": True,
                    "name": bat.name,
                    "status": status,
                    "power_now_w": power_w,
                }
                break
            except (OSError, ValueError):
                continue

    rapl_path = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    if rapl_path.exists():
        if os.access(rapl_path, os.R_OK):
            rapl_status = "available"
        else:
            rapl_status = "unavailable_nonroot"
    else:
        rapl_status = "not_present"

    return {"battery": battery, "intel_rapl": rapl_status}


def parse_pw_top_batch(text: str) -> list[dict[str, Any]]:
    """Parse one `pw-top -b -n 2` batch snapshot into node rows."""
    # Take the last snapshot block (first block often shows zeroed deltas)
    blocks = re.split(r"(?=^S\s+ID\s+QUANT)", text, flags=re.MULTILINE)
    block = blocks[-1] if blocks else text
    rows: list[dict[str, Any]] = []
    for line in block.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        parts = line_s.split()
        if len(parts) < 9 or parts[1] == "ID":
            continue
        state = parts[0]
        if state not in ("R", "S", "I", "C", "E"):
            continue
        if not parts[1].isdigit():
            continue
        rows.append(
            {
                "state": state,
                "id": int(parts[1]),
                "quant": _safe_int(parts[2]),
                "rate": _safe_int(parts[3]),
                "wait": parts[4],
                "busy": parts[5],
                "w_q": parts[6],
                "b_q": parts[7],
                "err": _safe_int(parts[8]),
                "name": " ".join(parts[9:]),
            }
        )
    return rows


def query_pipewire_top(timeout_sec: float = 3.0) -> dict[str, Any]:
    """Capture a short non-interactive `pw-top -b -n 2` snapshot if PipeWire is running."""
    if not shutil.which("pw-top"):
        return {"available": False, "reason": "pw-top not found"}
    try:
        res = subprocess.run(
            ["pw-top", "-b", "-n", "2"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if res.returncode != 0:
            return {"available": False, "reason": res.stderr.strip() or "pw-top failed"}
        rows = parse_pw_top_batch(res.stdout)
        running_rows = [r for r in rows if r["state"] == "R"]
        total_err = sum(r["err"] for r in rows)
        broadcast_rows = [r for r in rows if "nv_broadcast" in r["name"].lower()]
        return {
            "available": True,
            "node_count": len(rows),
            "running_nodes": len(running_rows),
            "total_xrun_errors": total_err,
            "broadcast_nodes": broadcast_rows,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)}


def count_device_open_fds(device_path: str) -> int:
    """Count open file descriptors owned by the current user pointing to `device_path`."""
    count = 0
    proc_root = Path("/proc")
    try:
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            fd_dir = entry / "fd"
            try:
                for fd in fd_dir.iterdir():
                    try:
                        if os.readlink(fd) == device_path:
                            count += 1
                    except OSError:
                        continue
            except OSError:
                continue
    except OSError:
        return 0
    return count


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute min/mean/p50/p95/p99/p999/max summary statistics."""
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p999": 0.0,
            "max": 0.0,
        }
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, int(round((n - 1) * p)))
        return s[idx]

    return {
        "count": n,
        "min": round(s[0], 3),
        "mean": round(sum(s) / n, 3),
        "p50": round(pct(0.50), 3),
        "p95": round(pct(0.95), 3),
        "p99": round(pct(0.99), 3),
        "p999": round(pct(0.999), 3),
        "max": round(s[-1], 3),
    }


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(text: str) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0
