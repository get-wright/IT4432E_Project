#!/usr/bin/env bash
# Runs ON the VM. Sets up Python env, Kaggle creds, downloads datasets.
set -euo pipefail

REPO_URL="${REPO_URL:-}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
KAGGLE_KEY="${KAGGLE_KEY:?set KAGGLE_KEY}"

cd "$HOME"

# 1. Clone repo if not present.
if [ ! -d IT4432E_Project ]; then
  test -n "$REPO_URL" || { echo "set REPO_URL"; exit 1; }
  git clone "$REPO_URL" IT4432E_Project
fi
cd IT4432E_Project

# 2. Python deps. Deep Learning VM has conda+PyTorch+CUDA preinstalled.
pip install -e ".[dev]"

# 3. Kaggle credentials.
mkdir -p "$HOME/.kaggle"
cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
chmod 600 "$HOME/.kaggle/kaggle.json"

# 4. Download datasets.
download() {
  local slug="$1" dest="$2"
  mkdir -p "$dest"
  if [ -z "$(ls -A "$dest")" ]; then
    kaggle datasets download -d "$slug" -p "$dest"
    (cd "$dest" && unzip -q -o "*.zip" && rm -f "*.zip")
  else
    echo "$dest already populated; skipping download."
  fi
}

download "debarghamitraroy/casia-webface" "preprocess-data/casia-webface"
download "jessicali9530/celeba-dataset"   "preprocess-data/celeba"
download "jessicali9530/lfw-dataset"      "preprocess-data/lfw"

echo "Setup complete."
