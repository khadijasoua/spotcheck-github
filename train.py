"""Fine-tune an ImageNet-pretrained ResNet-18 on DermaMNIST.

Set WEIGHTED to choose the variant:
    WEIGHTED = False -> plain cross-entropy        -> model_plain.pt
    WEIGHTED = True  -> inverse-frequency weights  -> model_weighted.pt

The checkpoint with the best validation balanced accuracy is kept.

Usage: python train.py
"""

import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from common import CLASS_NAMES, build_model, get_data, get_probs

WEIGHTED = True
EPOCHS = 5
MODEL_FILE = "model_weighted.pt" if WEIGHTED else "model_plain.pt"

torch.manual_seed(0)

train_set = get_data("train", train_mode=True)
val_set = get_data("val")
loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

model = build_model()

if WEIGHTED:
    # w_c = N / (K * N_c): rare classes contribute more to the loss.
    labels = torch.tensor(train_set.labels.flatten()).long()
    counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    print("Class weights:", weights)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
else:
    loss_fn = nn.CrossEntropyLoss()

# Small learning rate: refine the pretrained features rather than overwrite them.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in loader:
        y = y.squeeze(1).long()
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    probs, labels_val = get_probs(model, val_set)
    preds = probs.argmax(1)
    acc = (preds == labels_val).float().mean().item()
    bal = balanced_accuracy_score(labels_val, preds)
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss {total_loss / len(loader):.3f}  "
          f"val accuracy {acc:.1%}  balanced {bal:.1%}")

    # Select on balanced accuracy so the majority class can't dominate.
    if bal > best:
        best = bal
        torch.save(model.state_dict(), MODEL_FILE)

print("Saved the best model to", MODEL_FILE)
