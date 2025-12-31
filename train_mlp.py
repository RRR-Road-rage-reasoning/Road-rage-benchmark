import os
import glob
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score

# =====================================================
# 1. Reproducibility
# =====================================================
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 2. Feature Dataset
# =====================================================
class FeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted video features and multi-label annotations.
    Feature files are stored as .pt tensors.
    Label encoding is inferred from file names.
    """

    def __init__(self, feature_dir: str):
        self.paths = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
        assert len(self.paths) > 0, "No feature files found in the directory."

        self.labels = []
        for path in self.paths:
            filename = os.path.basename(path)
            # Example filename format: xxx_010.pt
            label_str = filename.split("_")[1].split(".")[0]
            label = torch.tensor([int(c) for c in label_str], dtype=torch.float32)
            self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        feature = torch.load(self.paths[idx]).float()
        label = self.labels[idx]
        return feature, label


# =====================================================
# 3. Stable MLP Classifier
# =====================================================
class StableMLP(nn.Module):
    """
    A lightweight and stable multilayer perceptron for scene classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# 4. Training and Evaluation
# =====================================================
def train_and_evaluate(
    feature_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda"
):
    """
    Train and evaluate the MLP classifier using fixed train-test split.
    Evaluation metrics include Accuracy and F1-score.
    """

    set_seed(42)

    dataset = FeatureDataset(feature_dir)
    sample_feature, _ = dataset[0]
    input_dim = sample_feature.shape[0]
    print(f"[Info] Feature dimension: {input_dim}")

    # Train-test split (80% / 20%)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, total_size - train_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = StableMLP(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    # ---------------- Training ----------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            features = torch.nan_to_num(features.to(device), nan=0.0)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {avg_loss:.4f}")

    # ---------------- Evaluation ----------------
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu()

            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = (all_probs > 0.5).astype(int)

    # =====================================================
    # Binary classification: any positive label vs all-zero
    # =====================================================
    y_true_binary = (all_labels.sum(axis=1) > 0).astype(int)
    y_pred_binary = (all_preds.sum(axis=1) > 0).astype(int)

    print("\n===== Binary Classification (Any-Positive vs All-Zero) =====")
    print("Accuracy:", accuracy_score(y_true_binary, y_pred_binary))
    print("F1-score:", f1_score(y_true_binary, y_pred_binary, zero_division=0))

    # =====================================================
    # All-zero class performance
    # =====================================================
    print("\n===== All-Zero Class Performance =====")
    zero_indices = (y_true_binary == 0)
    if zero_indices.sum() > 0:
        pred_zero = (all_preds[zero_indices].sum(axis=1) == 0).astype(int)
        true_zero = np.ones_like(pred_zero)
        print(
            f"Accuracy: {accuracy_score(true_zero, pred_zero) * 100:.2f}%, "
            f"F1-score: {f1_score(true_zero, pred_zero, zero_division=0):.4f}"
        )
    else:
        print("Warning: No all-zero samples in the test set.")

    # =====================================================
    # Multi-label evaluation on positive samples only
    # =====================================================
    positive_indices = (y_true_binary == 1)
    if positive_indices.sum() > 0:
        print("\n===== Multi-label Classification on Positive Samples =====")
        for i in range(3):
            acc = accuracy_score(
                all_labels[positive_indices, i],
                all_preds[positive_indices, i]
            )
            f1 = f1_score(
                all_labels[positive_indices, i],
                all_preds[positive_indices, i],
                zero_division=0
            )
            print(f"Label {i+1}: Acc = {acc*100:.2f}%, F1 = {f1:.4f}")


# =====================================================
# 5. Entry Point
# =====================================================
if __name__ == "__main__":
    FEATURE_DIR = "/path/to/extracted_features"  # anonymized path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_evaluate(FEATURE_DIR, device=DEVICE)
