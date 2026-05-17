# Face Recognition with Siamese Network

Siamese-network face recognition trained on CASIA-WebFace + CelebA, evaluated on LFW, served via a FastAPI web app that captures faces from the browser webcam.

See [design spec](docs/superpowers/specs/2026-05-17-face-recognition-siamese-design.md) for full context.

## Repo layout
- `preprocess-data/` — raw datasets + sanity-check script
- `process-data/` — MTCNN-aligned faces + analysis notebook
- `training-pipeline/` — model, loss, training entry, results notebook
- `evaluation/` — LFW 10-fold benchmark
- `application/` — FastAPI + browser-webcam UI
- `infra/` — GCP VM provisioning scripts

## Quickstart (after training)
```
pip install -e ".[dev]"
cd application && uvicorn backend.main:app --reload
# open http://localhost:8000
```

## Training (on GCP)
See `infra/README.md`.
