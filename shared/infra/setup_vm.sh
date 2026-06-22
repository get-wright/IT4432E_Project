#!/usr/bin/env bash
# Runs ON the VM. Installs Python env, PyTorch + CUDA wheels, Kaggle creds, datasets.
# Targets bare Ubuntu 22.04 with NVIDIA driver pre-installed (e.g. Vietnamese provider H100 VM).
set -euo pipefail

REPO_URL="${REPO_URL:-}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
KAGGLE_KEY="${KAGGLE_KEY:?set KAGGLE_KEY}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"

cd "$HOME"

# 1. System packages.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3-pip python3-venv unzip git tmux curl >/dev/null

# 2. Clone repo if not present.
if [ ! -d IT4432E_Project ]; then
  test -n "$REPO_URL" || { echo "set REPO_URL"; exit 1; }
  git clone "$REPO_URL" IT4432E_Project
fi
cd IT4432E_Project
git fetch origin
git checkout feat/face-recognition-siamese || git checkout main
git pull --ff-only origin "$(git rev-parse --abbrev-ref HEAD)" || true

# 3. Virtualenv + deps.
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel setuptools >/dev/null

# PyTorch + torchvision first (CUDA-specific wheel).
pip install --index-url "$CUDA_INDEX" torch torchvision

# Project deps (skip torch/torchvision via constraint to keep CUDA wheel).
pip install -e ".[dev]"

# Verify CUDA visible from PyTorch.
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

# 4. Kaggle credentials.
mkdir -p "$HOME/.kaggle"
cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
chmod 600 "$HOME/.kaggle/kaggle.json"

# 5. Download datasets.
download() {
  local slug="$1" dest="$2"
  mkdir -p "$dest"
  if [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "Downloading $slug -> $dest"
    kaggle datasets download -d "$slug" -p "$dest"
    (cd "$dest" && for z in *.zip; do unzip -q -o "$z" && rm -f "$z"; done)
  else
    echo "$dest already populated; skipping."
  fi
}

download "debarghamitraroy/casia-webface" "preprocess-data/casia-webface"
download "jessicali9530/celeba-dataset"   "preprocess-data/celeba"
download "jessicali9530/lfw-dataset"      "preprocess-data/lfw"

echo "Setup complete. Datasets in preprocess-data/."
