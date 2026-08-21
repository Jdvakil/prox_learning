#!/usr/bin/env bash
# One-shot Octo environment setup + smoke test. Run on the GPU machine.
# Creates an isolated JAX venv (never mixed with the molmospaces torch venv)
# and verifies that octo-small loads and sees the GPU.
set -euo pipefail

OCTO_DIR=${OCTO_DIR:-$HOME/octo}
OCTO_PIN=${OCTO_PIN:-241fb3514b7c}   # commit this integration was written against

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }

if [ ! -d "$OCTO_DIR" ]; then
    git clone https://github.com/octo-models/octo.git "$OCTO_DIR"
fi
cd "$OCTO_DIR"
git checkout "$OCTO_PIN" 2>/dev/null || echo "[setup] pin not found, staying on $(git rev-parse --short HEAD)"

uv venv .venv --python 3.10
uv pip install -p .venv/bin/python -e . "jax[cuda12]" flask tensorflow tensorflow-datasets opencv-python-headless h5py

echo "=== smoke: GPU + model load ==="
.venv/bin/python - <<'PY'
import jax
print("jax devices:", jax.devices())
assert any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in jax.devices()), \
    "JAX does not see the GPU — check CUDA driver / jax[cuda12] install"
from octo.model.octo_model import OctoModel
m = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
print("OCTO LOADED")
print("  observation tokenizers:", sorted(m.config["model"]["observation_tokenizers"].keys()))
print("  heads:", sorted(m.config["model"]["heads"].keys()))
PY
echo "=== setup complete ==="
echo "next: build the dataset (see octo_env/README.md), then finetune_proximity.py"
