import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

# Assuming process_data_multi_res is already implemented in elnino_prediction_simple.py
from elnino_prediction_simple import process_data_multi_res

# Define the 2D CNN architecture
class MultiResCNN(nn.Module):
    def __init__(self, input_channels_1=1, input_channels_k=1):
        super(MultiResCNN, self).__init__()
        # Convolutional layers for resolution 1
        self.conv1_res1 = nn.Conv2d(input_channels_1, 8, kernel_size=3, padding=1)
        self.conv2_res1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool_res1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layers for resolution k
        self.conv1_resk = nn.Conv2d(input_channels_k, 8, kernel_size=3, padding=1)
        self.conv2_resk = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool_resk = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (initialize with placeholder size, updated in forward pass)
        self.fc1 = nn.Linear(1, 128)  # Placeholder size; will adjust dynamically
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_res1, x_resk):
        # Process resolution 1 independently
        x1 = self.pool_res1(torch.relu(self.conv1_res1(x_res1)))
        x1 = self.pool_res1(torch.relu(self.conv2_res1(x1)))
        x1 = x1.reshape(x1.size(0), -1)  # Flatten

        # Process resolution k independently
        xk = self.pool_resk(torch.relu(self.conv1_resk(x_resk)))
        xk = self.pool_resk(torch.relu(self.conv2_resk(xk)))
        xk = xk.reshape(xk.size(0), -1)  # Flatten

        # Concatenate flattened outputs
        x = torch.cat([x1, xk], dim=1)

        # Dynamically adjust fully connected layer if needed
        if x.size(1) != self.fc1.in_features:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Define a function to train and evaluate the model
def train_and_evaluate(data_res1, data_resk, labels, model, criterion, optimizer, device, num_epochs=10):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {
        'in_sample_accuracy': [],
        'out_sample_accuracy': [],
        'in_sample_f1': [],
        'out_sample_f1': []
    }

    for train_index, test_index in kf.split(data_res1):
        X1_train, X1_test = data_res1[train_index], data_res1[test_index]
        Xk_train, Xk_test = data_resk[train_index], data_resk[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X1_train = torch.tensor(X1_train).float().to(device)
        X1_test = torch.tensor(X1_test).float().to(device)
        Xk_train = torch.tensor(Xk_train).float().to(device)
        Xk_test = torch.tensor(Xk_test).float().to(device)
        y_train = torch.tensor(y_train).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)

        train_loader = DataLoader(list(zip(X1_train, Xk_train, y_train)), batch_size=32, shuffle=True)
        test_loader = DataLoader(list(zip(X1_test, Xk_test, y_test)), batch_size=32, shuffle=False)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for inputs1, inputsk, targets in train_loader:
                inputs1, inputsk, targets = inputs1.to(device), inputsk.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs1, inputsk)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_train_pred = (model(X1_train, Xk_train).squeeze() > 0.5).cpu().numpy()
            y_test_pred = (model(X1_test, Xk_test).squeeze() > 0.5).cpu().numpy()

        # Calculate metrics
        metrics['in_sample_accuracy'].append(accuracy_score(y_train.cpu(), y_train_pred))
        metrics['out_sample_accuracy'].append(accuracy_score(y_test.cpu(), y_test_pred))
        metrics['in_sample_f1'].append(f1_score(y_train.cpu(), y_train_pred))
        metrics['out_sample_f1'].append(f1_score(y_test.cpu(), y_test_pred))

    return metrics


import joblib  # For saving model-related data
import os

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lead_times = range(1, 19)
resolutions = [1, 2, 3, 5, 6]

results = []
saliency_data = []

# Create directories to save data
os.makedirs("saliency_data", exist_ok=True)

for lead_time in lead_times:
    for k in resolutions:
        # Prepare data for the given lead time and resolutions
        data_res1, labels = process_data_multi_res(lead_time, 1)
        data_resk, _ = process_data_multi_res(lead_time, k)

        # Initialize the model, loss function, and optimizer
        model = MultiResCNN(input_channels_1=data_res1.shape[1], input_channels_k=data_resk.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train and evaluate the model
        metrics = train_and_evaluate(data_res1, data_resk, labels, model, criterion, optimizer, device)

        # Aggregate results
        results.append({
            'lead_time': lead_time,
            'resolution_k': k,
            'in_sample_accuracy': np.mean(metrics['in_sample_accuracy']),
            'out_sample_accuracy': np.mean(metrics['out_sample_accuracy']),
            'in_sample_f1': np.mean(metrics['in_sample_f1']),
            'out_sample_f1': np.mean(metrics['out_sample_f1'])
        })

        # Save information for saliency map computation
        saliency_info = {
            "lead_time": lead_time,
            "resolution_k": k,
            "inputs_res1": data_res1,  # Save the inputs for resolution 1
            "inputs_resk": data_resk,  # Save the inputs for resolution k
            "labels": labels,          # Ground-truth labels
            "model_state": model.state_dict(),  # Model state
            "device": str(device)               # Device info
        }
        saliency_data.append(saliency_info)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('elnino_results_multires.csv', index=False)

# Save saliency data using joblib
joblib.dump(saliency_data, "saliency_data/saliency_data.pkl")

