#!/usr/bin/env bash
# Run on VM. Launches training in tmux, follows log.
set -euo pipefail

cd "$HOME/IT4432E_Project"

SESSION="train"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "python -m training-pipeline.src.train --config training-pipeline/configs/train.yaml 2>&1 | tee training-pipeline/train.log"

echo "Training started in tmux session '$SESSION'."
echo "Attach: tmux attach -t $SESSION"
echo "Tail:   tail -f training-pipeline/train.log"
