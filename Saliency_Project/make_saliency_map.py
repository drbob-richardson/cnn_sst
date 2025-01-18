import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from elnino_prediction_simple import process_data_multi_res

# Define the simplified CNN architecture
class SingleResCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(SingleResCNN, self).__init__()
        # Convolutional layers for resolution 1
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(1, 128)  # Placeholder size; will adjust dynamically
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Process resolution 1
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Flatten

        # Dynamically adjust fully connected layer if needed
        if x.size(1) != self.fc1.in_features:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Function to compute and save saliency maps
def compute_and_save_saliency(model, inputs, labels, save_dir, lead_time, device):
    """
    Compute and save saliency maps for the single resolution model.

    Args:
        model: Trained PyTorch model.
        inputs: Input array (data_res1).
        labels: Ground-truth labels (NumPy array).
        save_dir: Directory to save the saliency plots.
        lead_time: Lead time for the data.
        device: PyTorch device (CPU or GPU).
    """
    model.eval()

    # Convert inputs and labels to PyTorch tensors
    data_res1 = torch.tensor(inputs, dtype=torch.float32, requires_grad=True).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    # Retain gradients for saliency computation
    data_res1.retain_grad()

    # Forward pass and compute loss
    outputs = model(data_res1)
    loss = nn.BCELoss()(outputs.squeeze(), labels)
    loss.backward()

    # Extract saliency map
    saliency = data_res1.grad.abs().squeeze().cpu().numpy()

    # Aggregate saliency if necessary (mean over samples)
    saliency = saliency.mean(axis=0)

    # Plot and save saliency map
    plt.figure(figsize=(8, 6))
    plt.title(f"Saliency Map (Lead Time: {lead_time})")
    plt.imshow(saliency, cmap="hot")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"saliency_lead_{lead_time}.png"))
    plt.close()


# Define training and evaluation function
def train_and_evaluate(data_res1, labels, model, criterion, optimizer, device, num_epochs=10):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {
        'in_sample_accuracy': [],
        'out_sample_accuracy': [],
        'in_sample_f1': [],
        'out_sample_f1': []
    }

    for train_index, test_index in kf.split(data_res1):
        X_train, X_test = data_res1[train_index], data_res1[test_index]
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

        # Evaluate the model
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
results = []

# Create directory for saliency plots
os.makedirs("saliency_plots", exist_ok=True)

for lead_time in lead_times:
    # Prepare data
    data_res1, labels = process_data_multi_res(lead_time, 1)

    # Initialize the model, loss function, and optimizer
    model = SingleResCNN(input_channels=data_res1.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    metrics = train_and_evaluate(data_res1, labels, model, criterion, optimizer, device)

    # Save metrics
    results.append({
        'lead_time': lead_time,
        'in_sample_accuracy': np.mean(metrics['in_sample_accuracy']),
        'out_sample_accuracy': np.mean(metrics['out_sample_accuracy']),
        'in_sample_f1': np.mean(metrics['in_sample_f1']),
        'out_sample_f1': np.mean(metrics['out_sample_f1'])
    })

    # Compute and save saliency map
    save_dir = os.path.join("saliency_plots", f"lead_{lead_time}")
    os.makedirs(save_dir, exist_ok=True)
    compute_and_save_saliency(model, data_res1, labels, save_dir, lead_time, device)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('elnino_results_single_res.csv', index=False)
