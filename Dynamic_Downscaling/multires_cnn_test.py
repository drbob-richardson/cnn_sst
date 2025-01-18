import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import csv
import os

# Load the dataset
file_path = "sst_data.nc"  # Path to the uploaded file
sst_broad = xr.open_dataset(file_path)['sst']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data processing function for multi-resolution input
def process_data_multi_res(lead_time, resolutions, seed=1975):
    torch.manual_seed(seed)

    # Define the Nino 3.4 region for label calculation
    nino34_lat_range = slice(-5, 5)
    nino34_lon_range = slice(190, 240)
    sst_nino34 = sst_broad.sel(lat=nino34_lat_range, lon=nino34_lon_range)

    # Resample to monthly means
    sst_broad_monthly = sst_broad.resample(time='M').mean()
    sst_nino34_monthly = sst_nino34.resample(time='M').mean()

    # Calculate anomalies
    sst_anomalies = sst_broad_monthly.groupby("time.month") - sst_broad_monthly.groupby("time.month").mean(dim="time")
    sst_nino34_anomalies = sst_nino34_monthly.groupby("time.month") - sst_nino34_monthly.groupby("time.month").mean(dim="time")

    # Compute labels for all time steps
    T = len(sst_nino34_anomalies['time'])
    labels = [
        1 if sst_nino34_anomalies.isel(time=t + lead_time).mean(dim=["lat", "lon"]).values > 0.5 else 0
        for t in range(T - lead_time)
    ]

    # Generate input data at multiple resolutions
    data_multi_res = []
    for n in resolutions:
        sst_downsampled = sst_anomalies.isel(lat=slice(0, None, n), lon=slice(0, None, n))
        data = np.array([
            sst_downsampled.isel(time=t).values for t in range(T - lead_time)
        ])
        data_multi_res.append(data)

    # Combine multi-resolution inputs
    combined_data = np.concatenate(data_multi_res, axis=-1)
    
    # Add a channel dimension for CNN compatibility
    combined_data = combined_data[:, np.newaxis, :, :]

    return combined_data, np.array(labels)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, input_height, input_width):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the flattened features dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, input_height, input_width)
            sample_output = self.pool(torch.relu(self.conv2(self.pool(torch.relu(self.conv1(sample_input))))))
            self.flattened_size = sample_output.view(-1).size(0)

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset class
class SSTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Training and evaluation function
def run_experiment(resolutions, lead_time=12, epochs=50, k_folds=5):
    data, labels = process_data_multi_res(lead_time, resolutions)
    dataset = SSTDataset(data, labels)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")
        train_data = torch.utils.data.Subset(dataset, train_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        # Define model, loss, and optimizer
        input_height, input_width = data.shape[2], data.shape[3]  # Extract height and width from data
        model = SimpleCNN(input_channels=len(resolutions), input_height=input_height, input_width=input_width).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = torch.sigmoid(model(inputs))
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())

        val_preds = (np.array(val_preds) > 0.5).astype(int)
        accuracy = accuracy_score(val_labels, val_preds)
        fold_accuracies.append(accuracy)

        print(f"Accuracy for fold {fold + 1}: {accuracy:.4f}")

    mean_accuracy = np.mean(fold_accuracies)
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    return mean_accuracy

# Run experiments for individual and combined resolutions
individual_resolutions = [1, 2, 3, 4, 5, 6, 8, 10]
combined_resolutions = individual_resolutions

results = {}

# Test individual resolutions
for res in individual_resolutions:
    print(f"Testing resolution: {res}")
    accuracy = run_experiment([res])
    results[f"Resolution {res}"] = accuracy

# Test combined resolutions
print("Testing combined resolutions")
accuracy = run_experiment(combined_resolutions)
results["Combined Resolutions"] = accuracy

# Save results to a CSV file
results_file = "multi_resolution_results.csv"
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Resolution", "Accuracy"])
    for res, acc in results.items():
        writer.writerow([res, acc])

print("Experiment complete. Results saved to", results_file)
