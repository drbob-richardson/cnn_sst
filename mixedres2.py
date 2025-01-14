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

        # Fully connected layers
        self.fc1 = nn.Linear(1, 128)  # Placeholder size; will adjust dynamically
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_res1, x_resk):
        # Reshape inputs if they have an extra dimension
        if len(x_res1.shape) == 5:  # If input has 5 dimensions
            x_res1 = x_res1.view(-1, x_res1.size(2), x_res1.size(3), x_res1.size(4))  # Combine batch and sequence dims
        if len(x_resk.shape) == 5:
            x_resk = x_resk.view(-1, x_resk.size(2), x_resk.size(3), x_resk.size(4))

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


# Function to compute and save saliency maps
def compute_and_save_saliency(model, inputs, labels, save_dir, lead_time, resolution_k, device):
    """
    Compute and save saliency maps for both resolutions (res1 and resk).

    Args:
        model: Trained PyTorch model.
        inputs: Tuple of input arrays (data_res1, data_resk).
        labels: Ground-truth labels (NumPy array).
        save_dir: Directory to save the saliency plots.
        lead_time: Lead time for the data.
        resolution_k: Resolution level k.
        device: PyTorch device (CPU or GPU).
    """
    model.eval()

    # Convert inputs and labels to PyTorch tensors
    data_res1, data_resk = inputs
    data_res1 = torch.tensor(data_res1, dtype=torch.float32, requires_grad=True).to(device)
    data_resk = torch.tensor(data_resk, dtype=torch.float32, requires_grad=True).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    # Retain gradients for saliency computation
    data_res1.retain_grad()
    data_resk.retain_grad()

    # Compute saliency for resolution 1
    outputs = model(data_res1, data_resk)
    loss = nn.BCELoss()(outputs.squeeze(), labels)
    loss.backward()

    # Extract saliency maps
    saliency_res1 = data_res1.grad.abs().squeeze().cpu().numpy()
    saliency_resk = data_resk.grad.abs().squeeze().cpu().numpy()

    # Plot and save saliency map for resolution 1
    plt.figure(figsize=(8, 6))
    plt.title(f"Saliency Map (Lead Time: {lead_time}, Resolution: 1)")
    plt.imshow(saliency_res1, cmap="hot")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"saliency_lead_{lead_time}_res_1.png"))
    plt.close()

    # Plot and save saliency map for resolution k
    plt.figure(figsize=(8, 6))
    plt.title(f"Saliency Map (Lead Time: {lead_time}, Resolution: {resolution_k})")
    plt.imshow(saliency_resk, cmap="hot")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"saliency_lead_{lead_time}_res_{resolution_k}.png"))
    plt.close()



# Define training and evaluation function
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

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lead_times = range(1, 19)
resolutions = [1, 2, 3, 5, 6]

results = []

# Create directory for saliency plots
os.makedirs("saliency_plots", exist_ok=True)

for lead_time in lead_times:
    for k in resolutions:
        # Prepare data
        data_res1, labels = process_data_multi_res(lead_time, 1)
        data_resk, _ = process_data_multi_res(lead_time, k)

        # Initialize the model, loss function, and optimizer
        model = MultiResCNN(input_channels_1=data_res1.shape[1], input_channels_k=data_resk.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train and evaluate
        metrics = train_and_evaluate(data_res1, data_resk, labels, model, criterion, optimizer, device)

        # Save metrics
        results.append({
            'lead_time': lead_time,
            'resolution_k': k,
            'in_sample_accuracy': np.mean(metrics['in_sample_accuracy']),
            'out_sample_accuracy': np.mean(metrics['out_sample_accuracy']),
            'in_sample_f1': np.mean(metrics['in_sample_f1']),
            'out_sample_f1': np.mean(metrics['out_sample_f1'])
        })

        # Compute and save saliency map
        save_dir = os.path.join("saliency_plots", f"lead_{lead_time}_res_{k}")
        os.makedirs(save_dir, exist_ok=True)
        compute_and_save_saliency(model, (data_res1, data_resk), labels, save_dir, lead_time, k, device)


# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('elnino_results_multires.csv', index=False)
