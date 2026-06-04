import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

# Assuming process_data_multi_res is already implemented in elnino_prediction_simple.py
from elnino_prediction_simple import process_data_multi_res

# Define the 2D CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, input_height=30, input_width=90):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)              # Second convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                     # Adjusted Max pooling layer
        self.dropout = nn.Dropout(p=0.5)

        # Dynamically calculate the flattened size based on input dimensions
        def calculate_flattened_size():
            x = torch.zeros((1, input_channels, input_height, input_width))
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            return x.numel()

        self.flattened_size = calculate_flattened_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.contiguous().view(-1, self.flattened_size)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Define a function to train and evaluate the model
def train_and_evaluate(data, labels, model, criterion, optimizer, device, num_epochs=100):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model_state = None
    best_out_sample_f1 = -float('inf')

    metrics = {
        'in_sample_accuracy': [],
        'out_sample_accuracy': [],
        'in_sample_f1': [],
        'out_sample_f1': []
    }

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train = torch.tensor(X_train).float().to(device)
        X_test = torch.tensor(X_test).float().to(device)
        y_train = torch.tensor(y_train).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)

        train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
        test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

            # Evaluate at the end of each epoch
            model.eval()
            with torch.no_grad():
                y_test_pred = (model(X_test).squeeze() > 0.5).cpu().numpy()
                current_out_sample_f1 = f1_score(y_test.cpu(), y_test_pred)

                # Save the best model
                if current_out_sample_f1 > best_out_sample_f1:
                    best_out_sample_f1 = current_out_sample_f1
                    best_model_state = model.state_dict()

        # Load the best model state
        model.load_state_dict(best_model_state)

        # Final evaluation with the best model
        model.eval()
        with torch.no_grad():
            y_train_pred = (model(X_train).squeeze() > 0.5).cpu().numpy()
            y_test_pred = (model(X_test).squeeze() > 0.5).cpu().numpy()

        # Calculate metrics
        metrics['in_sample_accuracy'].append(accuracy_score(y_train.cpu(), y_train_pred))
        metrics['out_sample_accuracy'].append(accuracy_score(y_test.cpu(), y_test_pred))
        metrics['in_sample_f1'].append(f1_score(y_train.cpu(), y_train_pred))
        metrics['out_sample_f1'].append(f1_score(y_test.cpu(), y_test_pred))

    return metrics

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lead_times = range(1, 19)
resolutions = [1, 2, 3, 5, 6]

results = []

for lead_time in lead_times:
    for resolution in resolutions:
        # Prepare data for the given lead time and resolution
        data, labels = process_data_multi_res(lead_time, resolution)

        # Initialize the model, loss function, and optimizer
        model = SimpleCNN(input_channels=data.shape[1], input_height=data.shape[2], input_width=data.shape[3]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train and evaluate the model
        metrics = train_and_evaluate(data, labels, model, criterion, optimizer, device)

        # Aggregate results
        results.append({
            'lead_time': lead_time,
            'resolution': resolution,
            'in_sample_accuracy': np.mean(metrics['in_sample_accuracy']),
            'out_sample_accuracy': np.mean(metrics['out_sample_accuracy']),
            'in_sample_f1': np.mean(metrics['in_sample_f1']),
            'out_sample_f1': np.mean(metrics['out_sample_f1'])
        })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('elnino_results.csv', index=False)
