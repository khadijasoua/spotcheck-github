"""Choose the melanoma referral threshold on the validation set, then check it once on the test set.

A lesion is referred when P(melanoma) >= t, instead of only when melanoma is the
most probable class. Lower t raises sensitivity at the cost of more false positives.

Usage: python threshold.py [model file]   (default: model_weighted.pt)
"""

import sys

import torch

from common import MEL, build_model, get_data, get_probs

MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "model_weighted.pt"
CHOSEN = 0.2  # selected from the validation results below

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_FILE))


def count(probs, labels, t):
    """Return (melanomas flagged, total melanomas, false positives) at threshold t."""
    flagged = probs[:, MEL] >= t
    is_mel = labels == MEL
    caught = (flagged & is_mel).sum().item()
    false_alarms = (flagged & ~is_mel).sum().item()
    return caught, is_mel.sum().item(), false_alarms


print("VALIDATION set: use this to choose your threshold")
probs, labels = get_probs(model, get_data("val"))
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    caught, total, fa = count(probs, labels, t)
    print(f"  threshold {t:.1f}: caught {caught}/{total} melanomas, {fa} false alarms")

# The test set is used once, after the threshold has been fixed.
print(f"\nTEST set with your chosen threshold {CHOSEN}")
probs, labels = get_probs(model, get_data("test"))
caught, total, fa = count(probs, labels, CHOSEN)
print(f"  caught {caught}/{total} melanomas ({caught / total:.1%}), {fa} false alarms")
