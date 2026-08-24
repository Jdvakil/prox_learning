#!/usr/bin/env bash
# One-shot Octo environment setup + smoke test. Run on the GPU machine.
#
# Every version below was pinned the hard way during the first bring-up
# (2026-08-24, pico): octo pins jax==0.4.20 in requirements.txt, and that
# era needs an old-style CUDA install plus several downgrades that newer
# releases silently broke. Do not "upgrade" these without re-testing:
#   jax[cuda12_pip]==0.4.20  - octo's pin; the cuda12 extra does not exist yet
#   nvidia-cudnn-cu12==8.9.* - jaxlib 0.4.20+cuda12.cudnn89 wants cuDNN 8.9,
#                              newer installs pull 9.x -> "cuDNN version 0"
#   scipy==1.11.4            - scipy 1.13 removed scipy.linalg.tril
#   numpy==1.26.4            - jax 0.4.20 predates numpy 2
#   transformers==4.38.2     - transformers 5 removed the Flax classes
#   tfds==4.9.3 / tf-metadata==1.14.0 / protobuf<5 - newer metadata needs
#                              protobuf 5, clashing with octo's tf 2.15
# Also uninstall jax-cuda12-plugin/pjrt if present: they belong to newer jax
# and break 0.4.20 with "cannot import name 'triton'".
set -euo pipefail

OCTO_DIR=${OCTO_DIR:-$HOME/octo}
OCTO_PIN=${OCTO_PIN:-241fb3514b7c}   # commit this integration was written against

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
UV=$(command -v uv || ls "$HOME/.cargo/bin/uv" "$HOME/.local/bin/uv" 2>/dev/null | head -1)

if [ ! -d "$OCTO_DIR" ]; then
    git clone https://github.com/octo-models/octo.git "$OCTO_DIR"
fi
cd "$OCTO_DIR"
git checkout "$OCTO_PIN" 2>/dev/null || echo "[setup] pin not found, staying on $(git rev-parse --short HEAD)"

"$UV" venv .venv --python 3.10
P=.venv/bin/python
"$UV" pip install -p "$P" -e . -r requirements.txt flask opencv-python-headless h5py
"$UV" pip uninstall -p "$P" jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true
"$UV" pip install -p "$P" "jax[cuda12_pip]==0.4.20" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
"$UV" pip install -p "$P" \
    "nvidia-cudnn-cu12==8.9.7.29" nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12
"$UV" pip install -p "$P" "scipy==1.11.4" "numpy==1.26.4" "transformers==4.38.2" \
    "tensorflow_datasets==4.9.3" "tensorflow-metadata==1.14.0" "protobuf<5"

# jaxlib finds the pip-installed CUDA libraries through LD_LIBRARY_PATH; every
# GPU invocation (server, finetune) must source this file first.
cat > "$OCTO_DIR/octo_env.sh" <<EOF
export LD_LIBRARY_PATH=\$(ls -d $OCTO_DIR/.venv/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
EOF

echo "=== smoke: GPU + model load ==="
source "$OCTO_DIR/octo_env.sh"
"$P" - <<'PY'
import jax
devs = jax.devices()
print("jax devices:", devs)
assert any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs), \
    "JAX does not see the GPU - check the nvidia-* pip packages and LD_LIBRARY_PATH"
from octo.model.octo_model import OctoModel
m = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
print("OCTO LOADED")
print("  observation tokenizers:", sorted(m.config["model"]["observation_tokenizers"].keys()))
print("  heads:", sorted(m.config["model"]["heads"].keys()))
PY
echo "=== setup complete ==="
echo "next: tfds build (see octo_env/README.md), then finetune_proximity.py"
echo "remember: source $OCTO_DIR/octo_env.sh before any GPU run"
