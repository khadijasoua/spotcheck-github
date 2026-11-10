"""Evaluate a trained model on the DermaMNIST test set.

Reports accuracy against the majority-class baseline, balanced accuracy,
melanoma recall and precision, a per-class report, and a confusion matrix.

Usage: python evaluate.py model_plain.pt     (writes model_plain_confusion.png)
       python evaluate.py model_weighted.pt
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, precision_score, recall_score)

from common import CLASS_NAMES, MEL, build_model, get_data, get_probs

MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "model_plain.pt"

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_FILE))

probs, labels = get_probs(model, get_data("test"))
preds = probs.argmax(1).numpy()
labels = labels.numpy()

# Majority-class baseline.
most_common = np.bincount(labels).argmax()
baseline = (labels == most_common).mean()

print("Model:", MODEL_FILE)
print(f"Baseline (always '{CLASS_NAMES[most_common]}'): {baseline:.1%}")
print(f"Accuracy:           {accuracy_score(labels, preds):.1%}")
print(f"Balanced accuracy:  {balanced_accuracy_score(labels, preds):.1%}")

# Melanoma as a binary task.
is_mel = labels == MEL
said_mel = preds == MEL
print(f"Melanoma recall:    {recall_score(is_mel, said_mel):.1%}  (melanomas caught)")
print(f"Melanoma precision: {precision_score(is_mel, said_mel, zero_division=0):.1%}  "
      "(melanoma alarms that were right)")
print()

print(classification_report(labels, preds, labels=range(len(CLASS_NAMES)),
                            target_names=CLASS_NAMES, digits=3, zero_division=0))

# Rows = true class, columns = predicted class.
short = [n[:15] for n in CLASS_NAMES]
ConfusionMatrixDisplay.from_predictions(labels, preds, labels=range(len(CLASS_NAMES)),
                                        display_labels=short, xticks_rotation=45, cmap="Blues")
plt.title(f"Confusion matrix: {MODEL_FILE}")
out = MODEL_FILE.replace(".pt", "_confusion.png")
plt.savefig(out, bbox_inches="tight")
print("Saved", out)
plt.show()
