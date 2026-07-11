#!/bin/bash
# installation/build_mnn_opencl.sh
# Build MNN + pymnn from source with the OpenCL (Mali-G610) backend and install
# it into the project venv, enabling the GPU-MNN inference mode on an RK3588.
# by fcascan 2026
#
# WHY THIS SCRIPT EXISTS
#   1. The PyPI `mnn` wheel is CPU-only (no OpenCL backend compiled in), so MNN
#      must be built from source with -DMNN_OPENCL=ON.
#   2. CRITICAL for the RK3588: recent MNN (>= 3.x) turns the Arm SME2 backend
#      and the KleidiAI microkernels ON BY DEFAULT for arm64. The RK3588
#      (Cortex-A76/A55) has NO SME2/SVE2/i8mm, and MNN's SME2 path is NOT gated
#      by runtime CPU detection -> a default build executes SME2 instructions
#      (smstart / fmopa / rdsvl) on a CPU that lacks them and the process dies
#      with "illegal hardware instruction" (SIGILL) the moment a GPU-MNN OpenCL
#      session is created (independent of model / precision / instance count).
#      This build therefore forces -DMNN_SME2=OFF -DMNN_KLEIDIAI=OFF so the
#      resulting binary contains no SME2 code. See README "GPU-MNN Inference".
#
# USAGE (run from the repo root, after ./setup.sh has created ./venv):
#   ./installation/build_mnn_opencl.sh [BUILD_ROOT]
#   BUILD_ROOT defaults to ~/mnn_build (the MNN source tree is kept there so the
#   auto-tuned OpenCL kernels and objects can be reused on a later rebuild).
set -euo pipefail

# --- locate repo root + project venv ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$REPO_ROOT/venv/bin/python"
BUILD_ROOT="${1:-$HOME/mnn_build}"

# MNN revision verified with this project (v3.6.0). Bump deliberately, but ALWAYS
# keep MNN_SME2=OFF / MNN_KLEIDIAI=OFF below regardless of the MNN version.
MNN_REPO="https://github.com/alibaba/MNN.git"
MNN_REF="4a1ac98"

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: project venv not found at $VENV_PY. Run ./setup.sh first." >&2
    exit 1
fi

echo "========================================"
echo " Build MNN + pymnn (OpenCL, SME2/KleidiAI OFF)"
echo "========================================"
echo "  repo root : $REPO_ROOT"
echo "  venv      : $VENV_PY"
echo "  build root: $BUILD_ROOT"
echo "  MNN ref   : $MNN_REF"
echo ""

# --- 1. build toolchain -----------------------------------------------------
echo "[1/6] Installing build dependencies (build-essential, cmake, git)..."
sudo apt update
sudo apt install -y build-essential cmake git

# --- 2. get the MNN source --------------------------------------------------
echo "[2/6] Fetching MNN source into $BUILD_ROOT/MNN ..."
mkdir -p "$BUILD_ROOT"
if [ ! -d "$BUILD_ROOT/MNN/.git" ]; then
    git clone "$MNN_REPO" "$BUILD_ROOT/MNN"
fi
cd "$BUILD_ROOT/MNN"
git fetch --all --tags || true
git checkout "$MNN_REF"

# --- 3. force SME2 + KleidiAI OFF in the pymnn build ------------------------
echo "[3/6] Patching pymnn build_deps.py: add -DMNN_SME2=OFF -DMNN_KLEIDIAI=OFF ..."
BD="pymnn/pip_package/build_deps.py"
if ! grep -q 'MNN_SME2=OFF' "$BD"; then
    sed -i "s|extra_opts = '-DMNN_LOW_MEMORY=ON'|extra_opts = '-DMNN_LOW_MEMORY=ON -DMNN_SME2=OFF -DMNN_KLEIDIAI=OFF'|" "$BD"
fi
grep -q 'MNN_SME2=OFF' "$BD" || { echo "ERROR: failed to patch $BD (unexpected build_deps.py layout)." >&2; exit 1; }
echo "  ok: $(grep -m1 'MNN_LOW_MEMORY=ON' "$BD" | sed 's/^ *//')"

# --- 4. build the static MNN libs (OpenCL) ----------------------------------
echo "[4/6] Building MNN static libs with OpenCL (~10-30 min on RK3588)..."
cd "$BUILD_ROOT/MNN/pymnn/pip_package"
CMAKE_POLICY_VERSION_MINIMUM=3.5 "$VENV_PY" build_deps.py opencl

# --- 5. build + install the pymnn extension into the venv -------------------
# rm -rf build/ is REQUIRED: distutils' build_ext only recompiles when the .cpp
# sources change, NOT when the static .a libs do. Without wiping build/, it
# copies the STALE cached _mnncengine*.so (still linked against the previous,
# SME2-enabled libs) and the SIGILL persists even though the libs were rebuilt.
echo "[5/6] Building + installing the pymnn extension into the venv..."
rm -rf build
"$VENV_PY" setup.py install --deps opencl

# --- 6. OpenCL loader symlink + verification --------------------------------
echo "[6/6] Provisioning libOpenCL.so symlink and verifying the build..."
# MNN's OpenCL loader dlopens a bare "libOpenCL.so"; this device ships only
# "libOpenCL.so.1". (The app also provisions this at startup when run as root.)
for d in /usr/lib/aarch64-linux-gnu /usr/lib; do
    if [ -e "$d/libOpenCL.so.1" ] && [ ! -e "$d/libOpenCL.so" ]; then
        sudo ln -sf "$d/libOpenCL.so.1" "$d/libOpenCL.so" && echo "  linked $d/libOpenCL.so -> libOpenCL.so.1"
    fi
done
sudo ldconfig || true

EXT="$("$VENV_PY" -c "import glob, os, sysconfig; sp=sysconfig.get_paths()['purelib']; print(glob.glob(os.path.join(sp, '_mnncengine*.so'))[0])")"
echo "  installed extension: $EXT"
if command -v objdump >/dev/null 2>&1; then
    SME="$(objdump -d "$EXT" 2>/dev/null | grep -iwcE 'smstart|smstop|rdsvl' || true)"
    echo "  SME2 instruction count in extension (MUST be 0): ${SME:-0}"
    if [ "${SME:-0}" != "0" ]; then
        echo "ERROR: extension still contains SME2 instructions -> it will SIGILL on the RK3588." >&2
        echo "       Ensure build_deps.py has -DMNN_SME2=OFF -DMNN_KLEIDIAI=OFF and that build/ was wiped." >&2
        exit 1
    fi
else
    echo "  (objdump not found; skipping SME2 static check)"
fi
"$VENV_PY" -c "import MNN; print('  MNN import OK')"

echo ""
echo "========================================"
echo " MNN OpenCL build installed successfully"
echo "========================================"
echo "Set 'inference_device = GPU-MNN' in config.ini to use the Mali-G610 path."
