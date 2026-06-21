# AdaFace

IResNet50 + AdaFace, trained externally (train_local.py, not in repo).
Drop the checkpoint at `application/models/adaface.pt`. Inference backbone: `iresnet.py`.
Eval: `evaluation/adaface_evaluation.ipynb` (8-suite) or `python -m evaluation.evaluate --model adaface`.
