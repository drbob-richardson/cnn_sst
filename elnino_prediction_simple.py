import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SSTDataset class
class SSTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Data processing function
def process_data_multi_res(lead_time, resolution=1, seed=1975):
    torch.manual_seed(seed)

    # Load the dataset and subset to the specified region
    file_path = "sst.mon.mean.nc"  # Updated file path
    sst_broad = xr.open_dataset(file_path)['sst'].sel(
        lat=slice(15,-15), lon=slice(170, 260)
    )

    # Define the label calculation region (Nino 3.4 region)
    nino34_lat_range = slice(5,-5)
    nino34_lon_range = slice(190, 240)
    sst_nino34 = sst_broad.sel(lat=nino34_lat_range, lon=nino34_lon_range)

    # Calculate anomalies
    sst_anomalies = sst_broad.groupby("time.month") - sst_broad.groupby("time.month").mean(dim="time")
    sst_nino34_anomalies = sst_nino34.groupby("time.month") - sst_nino34.groupby("time.month").mean(dim="time")

    # Compute labels for all time steps
    T = len(sst_nino34_anomalies['time'])
    labels = [
        1 if sst_nino34_anomalies.isel(time=t + lead_time).mean(dim=["lat", "lon"]).values > 0.5 else 0
        for t in range(T - lead_time)
    ]

    # Downsample data based on the resolution
    sst_downsampled = sst_anomalies.isel(lat=slice(0, None, resolution), lon=slice(0, None, resolution))
    data = np.array([
        sst_downsampled.isel(time=t).values for t in range(T - lead_time)
    ])

    # Add a channel dimension for CNN compatibility
    data = data[:, np.newaxis, :, :]

    return data, np.array(labels)

# Simplified CNN model definition with Dropout
class SimpleCNN(nn.Module):
    def __init__(self, input_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout after pooling layers
        self.dropout_conv = nn.Dropout2d(p=0.1)

        # Fully connected layers
        self.fc1 = nn.Linear(1, 16)  # Placeholder
        self.fc2 = nn.Linear(16, 1)

        # Dropout for fully connected layers
        self.dropout_fc = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout_conv(x)  # Apply dropout after pooling
        x = x.view(x.size(0), -1)  # Flatten
        if not hasattr(self, "flattened_size"):
            self.flattened_size = x.size(1)
            self.fc1 = nn.Linear(self.flattened_size, 16).to(x.device)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)  # Apply dropout after fc1
        x = self.fc2(x)
        return x

# Training and evaluation function
def run_experiment(lead_time=12, resolution=1, epochs=50, k_folds=5):
    data, labels = process_data_multi_res(lead_time, resolution)
    print("Input data shape:", data.shape)  # Debug: Verify data shape
    print("Labels shape:", labels.shape)

    input_channels = data.shape[1]  # Should be 1 (channel)
    dataset = SSTDataset(data, labels)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_model = None
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

            # Training loop
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluation loop
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = torch.sigmoid(model(inputs))
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())

            val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
            accuracy = accuracy_score(val_labels, val_preds_binary)
            precision = precision_score(val_labels, val_preds_binary, zero_division=1)
            recall = recall_score(val_labels, val_preds_binary, zero_division=1)
            f1 = f1_score(val_labels, val_preds_binary, zero_division=1)

            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1_scores.append(f1)

        # Aggregate metrics across folds
        mean_accuracy = np.mean(fold_accuracies)
        mean_precision = np.mean(fold_precisions)
        mean_recall = np.mean(fold_recalls)
        mean_f1 = np.mean(fold_f1_scores)

        print(f"Epoch {epoch + 1}: Accuracy={mean_accuracy:.4f}, Precision={mean_precision:.4f}, Recall={mean_recall:.4f}, F1 Score={mean_f1:.4f}")

        # Track the best model
        if mean_f1 > best_f1_score:
            best_f1_score = mean_f1
            best_model_state = model.state_dict()

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, "best_model_f1.pth")
        print(f"Best model saved with F1 Score={best_f1_score:.4f}")

# Main execution
if __name__ == "__main__":
    resolution = 3  # Modify as needed
    lead_time = 1  # Modify as needed
    run_experiment(lead_time=lead_time, resolution=resolution)
