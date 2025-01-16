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

# Define the deeper CNN architecture
class DeeperCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(DeeperCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Placeholder size; adjust dynamically
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Convolutional layers with batch normalization and ReLU
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        
        # Flatten the tensor
        x = x.reshape(x.size(0), -1)

        # Dynamically adjust fully connected layer if needed
        if x.size(1) != self.fc1.in_features:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)

        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Function to compute and save saliency maps
# (Unchanged, see original code)

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
# (Unchanged, see original code)

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lead_times = range(1, 19)
results = []

# Create directory for saliency plots
os.makedirs("saliency_plots2", exist_ok=True)

for lead_time in lead_times:
    # Prepare data
    data_res1, labels = process_data_multi_res(lead_time, 1)

    # Initialize the model, loss function, and optimizer
    model = DeeperCNN(input_channels=data_res1.shape[1]).to(device)
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
    save_dir = os.path.join("saliency_plots2", f"lead_{lead_time}")
    os.makedirs(save_dir, exist_ok=True)
    compute_and_save_saliency(model, data_res1, labels, save_dir, lead_time, device)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('elnino_results_deeper_cnn.csv', index=False)
