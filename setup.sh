#!/bin/bash
# setup.sh - Complete setup script for project
# by fcascan 2025

echo "======================================"
echo " Project Setup"
echo "======================================"

# Check if running as root for some operations
if [[ $EUID -eq 0 ]]; then
   echo "WARNING: Running as root. This script will install system packages."
else
   echo "INFO: Running as regular user. Some operations may require sudo."
fi

# Update package list
echo "Updating package list..."
sudo apt update

# Install Python3 and pip if not available
echo "Installing Python3 and pip..."
sudo apt install -y python3 python3-pip

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-opencv python3-numpy python3-psutil python3-pyudev

# Try to install via pip as well (for latest versions)
echo "Installing Python packages via pip..."
python3 -m pip install --user opencv-python numpy psutil pyudev

# Install RKNN toolkit if wheel file exists
WHEEL_FILE="installation/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
if [ -f "$WHEEL_FILE" ]; then
    echo "Installing RKNN toolkit from wheel file..."
    python3 -m pip install --user "$WHEEL_FILE"
else
    echo "WARNING: RKNN wheel file not found at $WHEEL_FILE"
    echo "NPU inference will not be available. CPU inference will be used instead."
fi

# Make scripts executable
chmod +x install_dependencies.py

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To run the program:"
echo "  sudo python3 main.py"
echo ""
echo "To test dependencies:"
echo "  python3 install_dependencies.py"
echo ""
echo "Configuration:"
echo "  Edit config.ini to change settings"
echo "  Switch between NPU and CPU inference in config.ini"
echo ""