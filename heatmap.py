"""Grad-CAM explanations for the class-weighted model.

explain(index) predicts one test image and returns its Grad-CAM overlay; app.py
uses it. Run directly, the script lists every melanoma missed at the referral
threshold and plots the first six with their heatmaps.

Usage: python heatmap.py   (writes heatmaps_melanoma.png)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import DermaMNIST
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from common import CLASS_NAMES, MEL, SIZE, build_model, to_tensor

MODEL_FILE = "model_weighted.pt"
THRESHOLD = 0.2  # referral threshold chosen in threshold.py

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

# Last convolutional block: 512 feature maps on a 4 x 4 grid for 128 x 128 input.
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

# Untransformed test images, for display.
test_raw = DermaMNIST(split="test", download=True, size=SIZE)


def explain(index):
    """Predict one test photo and make its heatmap.
    Returns: photo, photo with heatmap, true type, predicted type, probabilities."""
    img, label = test_raw[index]
    x = to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred = probs.argmax().item()

    # Explains the predicted class, not the threshold-based referral.
    gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]

    rgb = np.array(img, dtype=np.float32) / 255
    overlay = show_cam_on_image(rgb, gray, use_rgb=True)
    return img, overlay, int(label[0]), pred, probs


if __name__ == "__main__":
    # Melanomas whose probability falls below the referral threshold.
    all_mel = np.where(test_raw.labels.flatten() == MEL)[0]
    missed = []
    for i in all_mel:
        img, _ = test_raw[int(i)]
        with torch.no_grad():
            p_mel = torch.softmax(model(to_tensor(img).unsqueeze(0)), dim=1)[0, MEL].item()
        if p_mel < THRESHOLD:
            missed.append(int(i))
    print(f"Missed {len(missed)} melanomas. Photo numbers:", missed)
    mel_ids = missed[:6]  # use list(all_mel[:6]) to plot the first six melanomas instead

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    for col, i in enumerate(mel_ids):
        img, overlay, true, pred, probs = explain(int(i))
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"photo #{i}", fontsize=9)
        axes[1, col].imshow(overlay)
        result = "CAUGHT" if probs[MEL] >= THRESHOLD else "MISSED"
        axes[1, col].set_title(f"{result}  (melanoma {probs[MEL]:.0%})\n"
                               f"top answer: {CLASS_NAMES[pred][:16]}", fontsize=8.5)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("heatmaps_melanoma.png")
    plt.show()
