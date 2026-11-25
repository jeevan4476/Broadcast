#!/usr/bin/env bash
# Helper to install optional Linux dependencies for NVIDIA Broadcast (Linux)
# Usage:
#   ./install-deps.sh --fedora     # Install v4l2loopback, v4l-utils, RNNoise LADSPA via dnf
#   ./install-deps.sh --load-cam   # Load v4l2loopback virtual camera kernel module
#   ./install-deps.sh --venv       # Create Python 3.12 GPU AI venv (ONNX CUDA, OpenCV, MediaPipe, DeepFilterNet)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

show_help() {
    cat <<EOF
NVIDIA Broadcast (Linux) Dependency Setup Helper

Options:
  --fedora     Install Fedora system packages (akmod-v4l2loopback, v4l-utils, RNNoise LADSPA)
  --load-cam   Load v4l2loopback module as "NVIDIA Broadcast Camera" (/dev/video10)
  --venv       Create optional Python 3.12 GPU AI venv at ${VENV_DIR}
  --all        Run --fedora, --load-cam, and --venv
EOF
}

install_fedora() {
    echo "==> Installing Fedora packages (requires RPM Fusion for v4l2loopback)..."
    if ! sudo dnf install -y akmod-v4l2loopback v4l2loopback v4l-utils ladspa-noise-suppression-for-voice; then
        echo "Note: If akmod-v4l2loopback was not found, enable RPM Fusion Free first:"
        echo "  sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-\$(rpm -E %fedora).noarch.rpm"
        return 1
    fi
}

load_cam() {
    echo "==> Loading v4l2loopback virtual camera (/dev/video10)..."
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="NVIDIA Broadcast Camera" exclusive_caps=1
    echo "Loaded! Verify with: ${SCRIPT_DIR}/nv-broadcast devices --all"
}

install_venv() {
    echo "==> Creating Python 3.12 GPU AI venv at ${VENV_DIR}..."
    mkdir -p "$(dirname "${VENV_DIR}")"
    if command -v uv >/dev/null 2>&1 && command -v python3.12 >/dev/null 2>&1; then
        uv venv --python python3.12 --system-site-packages "${VENV_DIR}"
        uv pip install --python "${VENV_DIR}/bin/python" -r "${SCRIPT_DIR}/requirements.txt"
    elif command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv --system-site-packages "${VENV_DIR}"
        "${VENV_DIR}/bin/pip" install --upgrade pip
        "${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
    else
        echo "python3.12 not found; install python3.12 or use system Python fallback."
        return 1
    fi
    PYTHONPATH="${SCRIPT_DIR}" "${VENV_DIR}/bin/python" -m nv_broadcast.cli download-models
    echo "GPU AI venv and neural models ready at ${VENV_DIR}"
}

if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

for arg in "$@"; do
    case "${arg}" in
        --fedora) install_fedora ;;
        --load-cam) load_cam ;;
        --venv) install_venv ;;
        --all) install_fedora && load_cam && install_venv ;;
        -h|--help) show_help ;;
        *) echo "Unknown option: ${arg}"; show_help; exit 1 ;;
    esac
done
