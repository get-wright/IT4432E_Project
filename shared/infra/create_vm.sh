#!/usr/bin/env bash
# Create A100 VM + GCS bucket for training. Idempotent: skips existing resources.
set -euo pipefail

PROJECT="${PROJECT:-mk8s-sec-057aa9}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-siamese-train}"
BUCKET="${BUCKET:-${PROJECT}-siamese}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
ACCEL="${ACCEL:-type=nvidia-tesla-a100,count=1}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"

gcloud config set project "$PROJECT"

# 1. GCS bucket for checkpoints.
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  echo "Creating bucket gs://$BUCKET ..."
  gcloud storage buckets create "gs://$BUCKET" --location=us-central1
else
  echo "Bucket gs://$BUCKET exists."
fi

# 2. VM.
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  echo "VM $VM_NAME exists. Starting if stopped..."
  gcloud compute instances start "$VM_NAME" --zone="$ZONE" || true
else
  echo "Creating VM $VM_NAME ..."
  gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="$ACCEL" \
    --image-family="$IMAGE_FAMILY" \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform
fi

echo "Done. SSH: gcloud compute ssh $VM_NAME --zone=$ZONE"
