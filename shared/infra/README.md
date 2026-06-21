# Infra Runbook

## Prereqs
- `gcloud` authenticated to project `mk8s-sec-057aa9`.
- A100 quota in `us-central1` (fallback: change `MACHINE_TYPE` / `ACCEL` env vars to L4: `g2-standard-8` / `type=nvidia-l4,count=1`).
- Kaggle username + API key (passed via env, never committed).

## 1. Provision VM + bucket (laptop)
```
bash infra/create_vm.sh
```

## 2. SSH to VM, set up env (one-time)
```
gcloud compute ssh siamese-train --zone=us-central1-a
# on VM:
git clone <REPO_URL> IT4432E_Project
cd IT4432E_Project
export KAGGLE_USERNAME=n3m09999
export KAGGLE_KEY=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export REPO_URL=<REPO_URL>
bash infra/setup_vm.sh
```

## 3. Process datasets (one-time)
```
python process-data/process.py
```

## 4. Train
```
bash infra/run_training.sh
tmux attach -t train
```

Training auto-stops the VM on completion.

## 5. Pull checkpoint (laptop)
```
gcloud storage cp gs://mk8s-sec-057aa9-siamese/checkpoints/best.pt application/models/
```

## Cost guardrails
- VM is stopped automatically on training success/failure.
- To manually stop: `gcloud compute instances stop siamese-train --zone=us-central1-a`.
- To delete entirely: `gcloud compute instances delete siamese-train --zone=us-central1-a`.
