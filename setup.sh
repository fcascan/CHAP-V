#!/bin/bash
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
# setup.sh - Complete setup script for project

echo "======================================"
echo " Project Setup (Orange Pi / RK3588)"
echo "======================================"

# 1. Install system dependencies and Python 3.12
echo "Updating system and installing Python 3.12..."
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# 2. Create and activate virtual environment
echo "Creating virtual environment (venv)..."
python3.12 -m venv venv
source venv/bin/activate

# 3. Install requirements without sudo
echo "Installing base dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install RKNN
WHEEL_FILE="installation/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
if [ -f "$WHEEL_FILE" ]; then
    echo "Installing RKNN toolkit from local file..."
    pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_FILE"
else
    echo "WARNING: RKNN file not found at $WHEEL_FILE"
    echo "NPU acceleration will not be available. CPU only."
fi

# 5. Install librknnrt runtime library
LIB_FILE="installation/librknnrt.so"
LIB_URL="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
if [ -f "$LIB_FILE" ]; then
    echo "Installing librknnrt.so from local file..."
else
    echo "Downloading librknnrt.so from Rockchip..."
    mkdir -p installation
    if command -v curl >/dev/null 2>&1; then
        curl -L "$LIB_URL" -o "$LIB_FILE"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$LIB_FILE" "$LIB_URL"
    else
        echo "ERROR: Neither curl nor wget is available to download librknnrt.so"
        exit 1
    fi
fi

if [ -f "$LIB_FILE" ]; then
    echo "Copying librknnrt.so to /usr/lib/..."
    sudo install -m 755 "$LIB_FILE" /usr/lib/librknnrt.so
    sudo ldconfig
else
    echo "WARNING: librknnrt.so not found and could not be downloaded."
    echo "NPU runtime may fail unless the shared library is installed manually."
fi

# 6. Permissions (the installer lives in src/; invoked via the interpreter, not executed directly)
chmod +x src/install_dependencies.py 2>/dev/null || true

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "Configuration:"
echo "  Edit config.ini to change settings"
echo ""
echo "To run the program (console mode):"
echo "  sudo ./venv/bin/python main.py"
echo ""
echo "To run the web interface:"
echo "  sudo ./venv/bin/python main.py --web"
echo ""
