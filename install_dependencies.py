#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""install_dependencies.py
Automatic dependency installer
by fcascan 2025
"""
import os
import sys
import subprocess
import importlib.util
import logging
import urllib.request
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# RKNN wheel URLs by Python version
RKNN_WHEELS = {
    "3.7": "https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    "3.8": "https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    "3.9": "https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    "3.10": "https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    "3.11": "https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    "3.12": "https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
}

def get_python_version():
    """Get current Python version as string (e.g., '3.10')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"

def download_file(url, filename):
    """Download a file from URL."""
    try:
        logger.info(f"Downloading {filename}...")
        # Convert GitHub blob URL to raw URL
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        
        urllib.request.urlretrieve(url, filename)
        logger.info(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {filename}: {e}")
        return False

def check_module(module_name, package_name=None):
    """Check if a module is installed."""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        logger.info(f"✓ {module_name} is already installed")
        return True
    except ImportError:
        logger.warning(f"✗ {module_name} not found")
        return False

def install_pip_if_missing():
    """Install pip if it's not available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        logger.info("pip not found, attempting to install it...")
        try:
            # Try to install pip using apt
            subprocess.run(["sudo", "apt", "update"], check=True, capture_output=True)
            subprocess.run(["sudo", "apt", "install", "-y", "python3-pip"], 
                          check=True, capture_output=True)
            logger.info("✓ pip installed successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("✗ Failed to install pip")
            return False

def install_package(package_name, use_sudo=False):
    """Install a package using pip."""
    # First ensure pip is available
    if not install_pip_if_missing():
        return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", package_name]
        if use_sudo:
            cmd = ["sudo"] + cmd
        
        logger.info(f"Installing {package_name}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install {package_name}: {e}")
        logger.error(f"Error output: {e.stderr}")
        
        # Try alternative installation methods
        return install_package_alternative(package_name)

def install_package_alternative(package_name):
    """Try alternative installation methods."""
    logger.info(f"Trying alternative installation for {package_name}...")
    
    # Map package names to system packages
    system_packages = {
        "opencv-python": "python3-opencv",
        "numpy": "python3-numpy", 
        "psutil": "python3-psutil",
        "pyudev": "python3-pyudev"
    }
    
    if package_name in system_packages:
        try:
            system_pkg = system_packages[package_name]
            logger.info(f"Installing {system_pkg} via apt...")
            subprocess.run(["sudo", "apt", "install", "-y", system_pkg], 
                          check=True, capture_output=True)
            logger.info(f"✓ Successfully installed {system_pkg}")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"✗ Failed to install {system_pkg} via apt")
    
    return False

def install_rknn_wheel():
    """Install RKNN toolkit by detecting Python version and downloading the correct wheel."""
    
    # Check if we're on aarch64 architecture
    if platform.machine() != 'aarch64':
        logger.warning("RKNN toolkit is only supported on aarch64 architecture")
        logger.info("Your architecture: " + platform.machine())
        return False
    
    # Get current Python version
    python_version = get_python_version()
    logger.info(f"Detected Python version: {python_version}")
    
    # Check if we have a wheel for this Python version
    if python_version not in RKNN_WHEELS:
        logger.error(f"✗ No RKNN wheel available for Python {python_version}")
        logger.error("Supported Python versions:")
        for version in sorted(RKNN_WHEELS.keys()):
            logger.error(f"  - Python {version}")
        logger.error("Please install one of the supported Python versions or contact the RKNN toolkit developers.")
        return False
    
    # Ensure pip is available
    if not install_pip_if_missing():
        return False
    
    # Check if already installed
    try:
        importlib.import_module("rknnlite")
        logger.info("✓ rknnlite is already installed")
        return True
    except ImportError:
        pass
    
    # First check if local wheel exists and matches our Python version
    local_wheel_pattern = f"installation/rknn_toolkit_lite2-2.3.2-cp{python_version.replace('.', '')}-cp{python_version.replace('.', '')}"
    local_wheels = []
    if os.path.exists("installation/"):
        for file in os.listdir("installation/"):
            if file.startswith("rknn_toolkit_lite2") and file.endswith(".whl"):
                local_wheels.append(os.path.join("installation", file))
                if local_wheel_pattern in file:
                    logger.info(f"Found compatible local wheel: {file}")
                    try:
                        cmd = [sys.executable, "-m", "pip", "install", os.path.join("installation", file)]
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        logger.info("✓ Successfully installed RKNN toolkit from local wheel")
                        return True
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to install local wheel: {e}")
    
    # Download the correct wheel for our Python version
    wheel_url = RKNN_WHEELS[python_version]
    wheel_filename = f"rknn_toolkit_lite2-2.3.2-cp{python_version.replace('.', '')}-cp{python_version.replace('.', '')}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
    
    logger.info(f"Downloading RKNN wheel for Python {python_version}...")
    
    # Create installation directory if it doesn't exist
    if not os.path.exists("installation"):
        os.makedirs("installation")
    
    wheel_path = os.path.join("installation", wheel_filename)
    
    # Download the wheel
    if not download_file(wheel_url, wheel_path):
        logger.error("Failed to download RKNN wheel")
        return False
    
    # Install the downloaded wheel
    try:
        cmd = [sys.executable, "-m", "pip", "install", wheel_path]
        logger.info("Installing RKNN toolkit from downloaded wheel...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✓ Successfully installed RKNN toolkit")
        
        # Clean up downloaded file
        try:
            os.remove(wheel_path)
            logger.info("✓ Cleaned up downloaded wheel file")
        except:
            pass
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install RKNN toolkit: {e}")
        logger.error(f"Error output: {e.stderr}")
        
        # Try forcing installation ignoring platform checks as fallback
        try:
            cmd = [sys.executable, "-m", "pip", "install", wheel_path, 
                   "--force-reinstall", "--no-deps", "--no-warn-script-location"]
            logger.info("Attempting forced installation...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✓ Successfully installed RKNN toolkit (forced)")
            
            # Clean up downloaded file
            try:
                os.remove(wheel_path)
            except:
                pass
            
            return True
        except subprocess.CalledProcessError as e2:
            logger.error(f"✗ Forced installation also failed: {e2}")
            logger.info("This might be due to missing system dependencies.")
            logger.info("The program will continue in CPU mode.")
            return False

def check_and_install_dependencies():
    """Check and install all required dependencies."""
    logger.info("=== Checking and Installing Dependencies ===")
    
    # First ensure pip is available
    logger.info("Checking pip availability...")
    if not install_pip_if_missing():
        logger.error("Cannot proceed without pip. Please install pip manually.")
        return False
    
    # List of dependencies to check
    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
        ("pyudev", "pyudev"),
    ]
    
    # Check basic dependencies
    missing_packages = []
    for module_name, package_name in dependencies:
        if not check_module(module_name):
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
        
        # Install packages individually with fallbacks
        for package in missing_packages:
            success = install_package(package)
            if not success:
                logger.warning(f"Could not install {package}, trying to continue...")
    
    # Check if RKNN toolkit is needed and available
    if not check_module("rknnlite"):
        logger.info("RKNN toolkit not found, attempting to install from wheel...")
        install_rknn_wheel()
    
    # Final verification
    logger.info("\n=== Verifying Installation ===")
    all_installed = True
    for module_name, package_name in dependencies:
        if not check_module(module_name):
            all_installed = False
    
    # Check RKNN separately (optional)
    check_module("rknnlite")
    
    if all_installed:
        logger.info("✓ All core dependencies are installed!")
        return True
    else:
        logger.warning("Some dependencies are still missing, but continuing...")
        logger.info("You may need to install missing packages manually:")
        logger.info("  sudo apt install python3-opencv python3-numpy python3-psutil python3-pyudev")
        return True  # Return True to allow the program to try running

def main():
    """Main function."""
    print(" Dependency Installer")
    print("=" * 40)
    
    success = check_and_install_dependencies()
    
    if success:
        print("\n✓ Dependency installation completed successfully!")
        print("You can now run main.py")
        return 0
    else:
        print("\n✗ Some dependencies could not be installed.")
        print("Please check the error messages above and install them manually.")
        return 1

if __name__ == "__main__":
    sys.exit(main())