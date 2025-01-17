import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from elnino_prediction_simple import process_data_multi_res
from mixedres import MultiResCNN  # Assuming the mixed resolution model is saved in mixedres.py

# Function to compute and save saliency maps for mixed-resolution model
def compute_and_save_saliency_mixed_res(model, inputs_res1, inputs_resk, labels, save_dir, lead_time, resolution_k, device):
    """
    Compute and save saliency maps for the mixed-resolution model.

    Args:
        model: Trained PyTorch model.
        inputs_res1: Input array for resolution 1.
        inputs_resk: Input array for resolution k.
        labels: Ground-truth labels (NumPy array).
        save_dir: Directory to save the saliency plots.
        lead_time: Lead time for the data.
        resolution_k: The value of resolution k.
        device: PyTorch device (CPU or GPU).
    """
    model.eval()

    # Convert inputs and labels to PyTorch tensors
    data_res1 = torch.tensor(inputs_res1, dtype=torch.float32, requires_grad=True).to(device)
    data_resk = torch.tensor(inputs_resk, dtype=torch.float32, requires_grad=True).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    # Retain gradients for saliency computation
    data_res1.retain_grad()
    data_resk.retain_grad()

    # Forward pass and compute loss
    outputs = model(data_res1, data_resk)
    loss = nn.BCELoss()(outputs.squeeze(), labels)
    loss.backward()

    # Extract saliency maps
    saliency_res1 = data_res1.grad.abs().squeeze().cpu().numpy().mean(axis=0)
    saliency_resk = data_resk.grad.abs().squeeze().cpu().numpy().mean(axis=0)

    # Plot and save saliency maps
    for res, saliency, name in zip([1, resolution_k], [saliency_res1, saliency_resk], ["res1", f"res{resolution_k}"]):
        plt.figure(figsize=(8, 6))
        plt.title(f"Saliency Map (Lead Time: {lead_time}, Resolution: {name})")
        plt.imshow(saliency, cmap="hot")
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"saliency_lead_{lead_time}_{name}.png"))
        plt.close()

# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lead_times = range(1, 19)
resolutions = [1, 2, 3, 5, 6]

# Create directory for saliency plots
os.makedirs("saliency_plots_mixed", exist_ok=True)

for lead_time in lead_times:
    for k in resolutions:
        # Prepare data for the given lead time and resolutions
        data_res1, labels = process_data_multi_res(lead_time, 1)
        data_resk, _ = process_data_multi_res(lead_time, k)

        # Initialize the model
        model = MultiResCNN(input_channels_1=data_res1.shape[1], input_channels_k=data_resk.shape[1]).to(device)

        # Load pre-trained model weights if available (implement model saving/loading as needed)
        # model.load_state_dict(torch.load(f"model_lead_{lead_time}_res_{k}.pth"))

        # Compute and save saliency maps
        save_dir = os.path.join("saliency_plots_mixed", f"lead_{lead_time}_res_{k}")
        os.makedirs(save_dir, exist_ok=True)
        compute_and_save_saliency_mixed_res(model, data_res1, data_resk, labels, save_dir, lead_time, k, device)
