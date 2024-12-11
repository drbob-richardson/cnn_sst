import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import KFold

# Assuming process_data_multi_res is already implemented in elnino_prediction_simple.py
from elnino_prediction_simple import process_data_multi_res

# Define the 2D CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, input_height=30, input_width=90):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1)              # Second convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)                     # Max pooling layer
        self.dropout = nn.Dropout(p=0.5)

        # Dynamically calculate the flattened size based on input dimensions
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, input_height, input_width)
            sample_output = self._forward_features(sample_input)
            self.flattened_size = sample_output.view(-1).size(0)

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First conv-pool
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv-pool
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)  # Logits for BCEWithLogitsLoss
        return x

# Define the Dataset class
class SSTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Function to train and evaluate

def train_and_evaluate(lead_time=12, resolution=1, epochs=50):
    data, labels = process_data_multi_res(lead_time=lead_time, resolution=resolution)

    # Ensure labels are of type float for BCEWithLogitsLoss
    labels = labels.astype(np.float32)

    dataset = SSTDataset(data, labels)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model = None
    best_f1_score = 0.0

    for train_idx, val_idx in kf.split(dataset):
        train_data = torch.utils.data.Subset(dataset, train_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        model = SimpleCNN(input_channels=data.shape[1], input_height=data.shape[2], input_width=data.shape[3])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))  # Match target shape with output
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = torch.sigmoid(model(inputs))
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Compute F1 Score
            val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
            f1 = f1_score(val_labels, val_preds_binary)

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model.state_dict()

    # Save the best model
    if best_model:
        torch.save(best_model, "best_cnn_model.pth")
        print(f"Best model saved with F1 Score: {best_f1_score:.4f}")

# Run the training
if __name__ == "__main__":
    train_and_evaluate(lead_time=12, resolution=2, epochs=50)

