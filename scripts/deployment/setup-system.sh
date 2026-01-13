#!/bin/bash
# setup-system.sh - Install system-level dependencies (requires sudo)
#
# Usage: ./scripts/deployment/setup-system.sh
#
# These packages cannot be managed by aqua (shared libraries, headers, drivers)

set -euo pipefail

echo "=== GPU Audio Upsampler: System Dependencies ==="
echo ""

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Detect distribution
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "Error: Cannot detect Linux distribution"
    exit 1
fi

echo "Detected: $DISTRO"
echo ""

case $DISTRO in
    ubuntu|debian)
        echo "Installing packages via apt..."
        $SUDO apt update
        $SUDO apt install -y \
            build-essential \
            pkg-config \
            nvidia-cuda-toolkit \
            libsndfile1-dev \
            libasound2-dev \
            git
        ;;
    fedora)
        echo "Installing packages via dnf..."
        $SUDO dnf install -y \
            gcc-c++ \
            pkgconfig \
            cuda \
            libsndfile-devel \
            alsa-lib-devel \
            git
        ;;
    arch|manjaro)
        echo "Installing packages via pacman..."
        $SUDO pacman -Syu --needed \
            base-devel \
            pkgconf \
            cuda \
            libsndfile \
            alsa-lib \
            git
        ;;
    *)
        echo "Unsupported distribution: $DISTRO"
        echo ""
        echo "Please install the following packages manually:"
        echo "  - C++ compiler (g++ or clang++)"
        echo "  - pkg-config"
        echo "  - CUDA toolkit (nvcc, cuFFT)"
        echo "  - libsndfile development files"
        echo "  - ALSA development files"
        echo "  - git"
        exit 1
        ;;
esac

echo ""
echo "=== System dependencies installed ==="
echo ""
echo "Next steps:"
echo "  1. Install aqua: curl -sSfL https://raw.githubusercontent.com/aquaproj/aqua-installer/v4.0.4/aqua-installer | bash"
echo "  2. Add aqua to PATH (see aqua docs)"
echo "  3. Run: aqua i"
echo "  4. Run: uv sync"
echo "  5. Build: cmake -B build && cmake --build build -j\$(nproc)"
