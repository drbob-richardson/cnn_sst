"""Train a single-resolution CNN and checkpoint the best-F1 model.

Originally the ``__main__`` of ``elnino_prediction_simple.py``. The shared data
pipeline now lives in :mod:`src.data`; this script keeps the model definition and
the K-fold training loop that saves ``models/best_model_f1.pth``.

Run from the repository root:

    python experiments/train_f1_model.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SSTDataset, process_data_multi_res  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).resolve().parents[1]


class SimpleCNN(nn.Module):
    """Two-conv CNN with a lazily-sized classifier head."""

    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(1, 16)  # Placeholder; resized on first forward.
        self.fc2 = nn.Linear(16, 1)
        self.dropout_fc = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        if not hasattr(self, "flattened_size"):
            self.flattened_size = x.size(1)
            self.fc1 = nn.Linear(self.flattened_size, 16).to(x.device)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


def run_experiment(lead_time=12, resolution=1, epochs=50, k_folds=5):
    data, labels = process_data_multi_res(lead_time, resolution)
    print("Input data shape:", data.shape)
    print("Labels shape:", labels.shape)

    input_channels = data.shape[1]
    dataset = SSTDataset(data, labels)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_f1_score = 0.0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        fold_accuracies, fold_precisions, fold_recalls, fold_f1_scores = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_data = torch.utils.data.Subset(dataset, train_idx)
            val_data = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

            model = SimpleCNN(input_channels=input_channels).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for inputs, labels_batch in train_loader:
                inputs, labels_batch = inputs.to(device), labels_batch.to(device).view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels_batch in val_loader:
                    inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                    outputs = torch.sigmoid(model(inputs))
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_labels.extend(labels_batch.cpu().numpy().flatten())

            val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
            fold_accuracies.append(accuracy_score(val_labels, val_preds_binary))
            fold_precisions.append(precision_score(val_labels, val_preds_binary, zero_division=1))
            fold_recalls.append(recall_score(val_labels, val_preds_binary, zero_division=1))
            fold_f1_scores.append(f1_score(val_labels, val_preds_binary, zero_division=1))

        mean_f1 = np.mean(fold_f1_scores)
        print(
            f"Epoch {epoch + 1}: Accuracy={np.mean(fold_accuracies):.4f}, "
            f"Precision={np.mean(fold_precisions):.4f}, Recall={np.mean(fold_recalls):.4f}, "
            f"F1 Score={mean_f1:.4f}"
        )

        if mean_f1 > best_f1_score:
            best_f1_score = mean_f1
            best_model_state = model.state_dict()

    if best_model_state is not None:
        out_path = REPO_ROOT / "models" / "best_model_f1.pth"
        torch.save(best_model_state, out_path)
        print(f"Best model saved to {out_path} with F1 Score={best_f1_score:.4f}")


if __name__ == "__main__":
    run_experiment(lead_time=1, resolution=3)
