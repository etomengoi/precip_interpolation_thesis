#!/usr/bin/env bash
# One-shot GPU setup for Vast.ai (or any CUDA 12 Linux instance).
# Run once after creating an instance:
#   bash scripts/setup_gpu.sh
#
# Does:
#   1. apt deps (compilers, boost — needed for LightGBM CUDA build)
#   2. pip install -e ".[gpu]"  (overlays JAX with CUDA 12)
#   3. LightGBM rebuild with CUDA backend (replaces CPU wheel from step 2)
#   4. verifies JAX device + bayesnf + lightgbm imports
set -e

PYTHON=${PYTHON:-/app/.venv/bin/python}

echo "==> [1/4] System deps (apt)..."
apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake ninja-build git \
    libboost-dev libboost-system-dev libboost-filesystem-dev

echo "==> [2/4] Python deps from pyproject + JAX CUDA 12..."
$PYTHON -m pip install -e ".[gpu]"

echo "==> [3/4] Rebuilding LightGBM with CUDA backend..."
# CUDA backend (device='cuda') replaces the legacy OpenCL ('--gpu') backend,
# which hangs on RTX 4000-series during kernel JIT (LightGBM issue #5536).
$PYTHON -m pip uninstall lightgbm -y
git clone --recursive --depth 1 --branch v4.5.0 \
    https://github.com/microsoft/LightGBM /tmp/LightGBM
(cd /tmp/LightGBM && sh ./build-python.sh install --cuda)
rm -rf /tmp/LightGBM

echo "==> [4/4] Verifying..."
$PYTHON - <<'EOF'
import jax
import bayesnf
import lightgbm as lgb
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
print(f"BayesNF: {bayesnf.__version__ if hasattr(bayesnf, '__version__') else 'imported OK'}")
print(f"LightGBM: {lgb.__version__}")
# LightGBM CUDA smoke test
import numpy as np
lgb.train(
    {'device_type': 'cuda', 'objective': 'regression', 'num_leaves': 4, 'verbose': -1},
    lgb.Dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), label=np.array([1.0, 2.0])),
    num_boost_round=1,
)
print("LightGBM CUDA OK")
EOF

echo "==> Done."
