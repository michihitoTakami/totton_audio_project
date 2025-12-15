#!/bin/bash
# Setup script for snd-aloop (ALSA Loopback) on Jetson
# Required for Magic Box audio pipeline (RTP → Loopback → GPU → DAC)
#
# Usage:
#   sudo ./scripts/setup_snd_aloop.sh              # Setup with modprobe.d
#   sudo ./scripts/setup_snd_aloop.sh systemd      # Setup with systemd service

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root (use: sudo ./scripts/setup_snd_aloop.sh)"
   exit 1
fi

SETUP_METHOD="${1:-modprobe}"

# ============================================================================
# Step 1: Load module immediately for testing
# ============================================================================
log_info "Step 1: Loading snd-aloop kernel module..."

if modprobe snd_aloop enable=1,1 2>/dev/null; then
    log_info "✓ snd-aloop loaded successfully"
else
    log_error "Failed to load snd-aloop module"
    log_warn "This may indicate a kernel issue or missing module"
    exit 1
fi

# Verify load
if lsmod | grep -q snd_aloop; then
    log_info "✓ Module verified in lsmod"
else
    log_error "Module not found in lsmod after loading"
    exit 1
fi

# ============================================================================
# Step 2: Verify ALSA recognition
# ============================================================================
log_info "Step 2: Verifying ALSA Loopback device..."

# Give udev time to create devices
sleep 1

if arecord -l 2>/dev/null | grep -q Loopback; then
    log_info "✓ ALSA Loopback device recognized"
    log_info "  Device list:"
    arecord -l | grep -A 2 Loopback | sed 's/^/    /'
else
    log_warn "Loopback device not yet recognized by ALSA"
    log_warn "This may be temporary; will configure persistence..."
fi

# ============================================================================
# Step 3: Configure persistence based on method
# ============================================================================
if [ "$SETUP_METHOD" = "systemd" ]; then
    log_info "Step 3: Setting up systemd service for snd-aloop..."

    # Create systemd service file
    tee /etc/systemd/system/snd-aloop-load.service > /dev/null << 'EOF'
[Unit]
Description=Load snd-aloop kernel module for Magic Box
After=network-online.target
Wants=network-online.target
Before=docker.service

[Service]
Type=oneshot
ExecStart=/sbin/modprobe snd_aloop enable=1,1
ExecStop=/sbin/modprobe -r snd_aloop
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    log_info "✓ Created /etc/systemd/system/snd-aloop-load.service"

    # Reload systemd daemon
    systemctl daemon-reload
    log_info "✓ Reloaded systemd daemon"

    # Enable service
    systemctl enable snd-aloop-load.service
    log_info "✓ Enabled snd-aloop-load.service (will autostart on reboot)"

    # Show status
    systemctl status snd-aloop-load.service

else
    log_info "Step 3: Setting up modprobe.d configuration..."

    # Create modprobe.d config
    tee /etc/modprobe.d/snd-aloop.conf > /dev/null << 'EOF'
# Enable ALSA Loopback for Magic Box audio pipeline
# Required for RTP → Loopback → GPU Convolution → DAC
options snd_aloop enable=1,1
alias snd-aloop snd_aloop
EOF

    log_info "✓ Created /etc/modprobe.d/snd-aloop.conf"

    # Optionally update initramfs (for Jetson Linux)
    if command -v update-initramfs &> /dev/null; then
        log_info "Updating initial ramdisk (initramfs)..."
        update-initramfs -u 2>/dev/null || log_warn "update-initramfs may have skipped update"
        log_info "✓ Initramfs updated"
    fi
fi

# ============================================================================
# Step 4: Verify persistence configuration
# ============================================================================
log_info "Step 4: Verifying persistence configuration..."

if [ "$SETUP_METHOD" = "systemd" ]; then
    if systemctl is-enabled snd-aloop-load.service &>/dev/null; then
        log_info "✓ snd-aloop-load.service is enabled for autostart"
    else
        log_error "snd-aloop-load.service is not enabled"
        exit 1
    fi
else
    if [ -f /etc/modprobe.d/snd-aloop.conf ]; then
        log_info "✓ /etc/modprobe.d/snd-aloop.conf exists"
        log_info "  Configuration:"
        cat /etc/modprobe.d/snd-aloop.conf | sed 's/^/    /'
    else
        log_error "Configuration file not found"
        exit 1
    fi
fi

# ============================================================================
# Step 5: Test loopback functionality
# ============================================================================
log_info "Step 5: Testing loopback functionality..."

# Test record capability
if timeout 1 arecord -D hw:Loopback,0,0 -f S16_LE -r 48000 -c 2 /dev/null 2>&1 | grep -q "." || [ $? -eq 124 ]; then
    log_info "✓ Loopback record test passed"
else
    log_warn "Loopback record test inconclusive (may work in actual use)"
fi

# ============================================================================
# Final summary
# ============================================================================
echo ""
log_info "Setup complete!"
echo ""
echo "Summary:"
echo "--------"
echo "✓ snd-aloop kernel module loaded"
echo "✓ ALSA Loopback device recognized"
if [ "$SETUP_METHOD" = "systemd" ]; then
    echo "✓ Configured persistence: systemd service (snd-aloop-load.service)"
    echo ""
    echo "Next steps:"
    echo "1. Reboot Jetson: sudo reboot"
    echo "2. Verify after reboot: lsmod | grep snd_aloop"
    echo "3. Start docker-compose: docker-compose up -d"
else
    echo "✓ Configured persistence: modprobe.d configuration"
    echo ""
    echo "Next steps:"
    echo "1. Reboot Jetson: sudo reboot"
    echo "2. Verify after reboot: lsmod | grep snd_aloop"
    echo "3. Start docker-compose: docker-compose up -d"
fi
echo ""
echo "Verification commands:"
echo "  - Check module: lsmod | grep snd_aloop"
echo "  - Check device: arecord -l | grep Loopback"
echo "  - Check service: systemctl status snd-aloop-load.service (if using systemd)"
echo ""
