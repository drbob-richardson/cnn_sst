# simulate_structured_spatial_data.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import accuracy_score, f1_score
from captum.attr import Saliency

# --------------------- Config ----------------------------------------
N = 500        # number of samples
H, W = 24, 48  # image size
OUTDIR = "sim_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------- Region Definitions ----------------------------
def make_circle_mask(H, W, center, radius):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return ((yy - center[0])**2 + (xx - center[1])**2 <= radius**2).astype(np.float32)

region_A = make_circle_mask(H, W, center=(10, 15), radius=4)
region_B = make_circle_mask(H, W, center=(14, 30), radius=4)
region_distractor_1 = make_circle_mask(H, W, center=(5, 35), radius=3)
region_distractor_2 = make_circle_mask(H, W, center=(18, 10), radius=3)

# --------------------- Simulate Inputs -------------------------------
def simulate_spatial_input(N, H, W, signal_regions):
    X = np.random.randn(N, 1, H, W) * 0.5  # base noise
    for region in signal_regions:
        mask = region['mask']
        weight = region['weight']
        coef = np.random.randn(N, 1, 1, 1)  # per-sample variation
        X += coef * weight * mask[None, None, :, :]
    return X.astype(np.float32)

signal_regions = [
    {'mask': region_A, 'weight': 1.0},
    {'mask': region_B, 'weight': -1.0},
    {'mask': region_distractor_1, 'weight': 0.8},
    {'mask': region_distractor_2, 'weight': -0.8},
]


# ---------------------- Generate Labels ------------------------------
def generate_labels(X, region_A, region_B, threshold=0.5):
    A_vals = (X * region_A[None, None, :, :]).sum(axis=(2, 3)) / region_A.sum()
    B_vals = (X * region_B[None, None, :, :]).sum(axis=(2, 3)) / region_B.sum()
    A_active = (A_vals > threshold).squeeze()
    B_active = (B_vals > threshold).squeeze()
    y = (A_active ^ B_active).astype(np.float32)  # XOR
    score = (A_active + B_active).astype(np.float32)
    return y, score

# --------------------- Simple CNN Model -----------------------------
class PixelMaskGate(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.z_main = nn.Parameter(torch.zeros(H * W))
        self.tau = 0.5
        self.H, self.W = H, W
        self.mask = None

    def forward(self, x):
        m = torch.sigmoid(self.z_main / self.tau)
        self.mask = m.view(1, 1, self.H, self.W).to(x.device)
        return x * self.mask, m

class SmallCNN(nn.Module):
    def __init__(self, gate=None):
        super().__init__()
        self.gate = gate
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        if self.gate is not None:
            x, _ = self.gate(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ------------------ Train CNN Model ---------------------------------
def train_model(model, loader, epochs=60, gate=None, lambda_sp=1e-1, lambda_tv=5e-2):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        if gate is not None:
            gate.tau = 2 - 1.9 * (epoch / epochs)
        for xb, yb in loader:
            xb, yb = xb.float(), yb.view(-1, 1).float()
            logits = model(xb)
            loss = loss_fn(logits, yb)

            if gate is not None:
                _, m = gate(xb)
                m_flat = m.view(-1)
                tv_penalty = 0.0
                for i in range(H):
                    for j in range(W):
                        idx = i * W + j
                        if i + 1 < H:
                            tv_penalty += (m_flat[idx] - m_flat[idx + W]).abs()
                        if j + 1 < W:
                            tv_penalty += (m_flat[idx] - m_flat[idx + 1]).abs()
                tv_penalty /= (H * W)
                loss += lambda_sp * (m**2).mean() + lambda_tv * tv_penalty

            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

# ------------------ Evaluation & Timing Utilities -------------------
def evaluate_accuracy(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.float()).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return accuracy_score(all_labels.numpy(), all_preds.numpy())

def time_forward_pass(model, input_tensor, n_repeats=10):
    times = []
    for _ in range(n_repeats):
        start = time.time()
        _ = model(input_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - start)
    return np.mean(times)

def time_attribution(method, model, input_tensor, n_repeats=3):
    times = []
    baseline = torch.zeros_like(input_tensor)
    for _ in range(n_repeats):
        input_clone = input_tensor.clone().detach().requires_grad_(True)
        start = time.time()
        
        # Explicit type name check in case isinstance fails due to wrapping
        method_name = method.__class__.__name__

        if method_name == "GradientShap":
            _ = method.attribute(input_clone, baselines=baseline)
        elif method_name in ["IntegratedGradients", "DeepLift"]:
            _ = method.attribute(input_clone, baseline)
        else:
            _ = method.attribute(input_clone)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append(time.time() - start)
    return np.mean(times)

def eval_metrics(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.float()).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
    f1 = f1_score(all_labels.numpy(), all_preds.numpy())
    return acc, f1


def compute_interpretability_metrics(attr_map, region_A, region_B, region_d1, region_d2, top_percentile=90):
    """
    Compute IoU and saliency mass for true and distractor regions.

    Args:
        attr_map (np.ndarray): Attribution map (H, W)
        region_* (np.ndarray): Binary masks for true/distractor regions
        top_percentile (float): Top-k% threshold for binary IoU

    Returns:
        dict with IoU and saliency mass for signal and distractor
    """
    attr_flat = attr_map.flatten()
    threshold = np.percentile(attr_flat, top_percentile)
    binary_mask = (attr_map >= threshold).astype(np.float32)

    true_mask = np.logical_or(region_A, region_B).astype(np.float32)
    distractor_mask = np.logical_or(region_d1, region_d2).astype(np.float32)

    # IoU
    iou_true = np.sum(binary_mask * true_mask) / np.sum((binary_mask + true_mask) > 0)
    iou_distractor = np.sum(binary_mask * distractor_mask) / np.sum((binary_mask + distractor_mask) > 0)

    # Saliency mass
    total_mass = np.sum(attr_map) + 1e-8
    mass_true = np.sum(attr_map * true_mask) / total_mass
    mass_distractor = np.sum(attr_map * distractor_mask) / total_mass

    return {
        "iou_true": iou_true,
        "iou_distractor": iou_distractor,
        "mass_true": mass_true,
        "mass_distractor": mass_distractor
    }

def run_single_simulation(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate new data
    X_np = simulate_spatial_input(N, H, W, signal_regions)
    y_np, _ = generate_labels(X_np, region_A, region_B)

    X = torch.tensor(X_np)
    y = torch.tensor(y_np)

    # Train/test split
    ds = TensorDataset(X, y)
    train_size = int(0.8 * N)
    train_ds, test_ds = random_split(ds, [train_size, N - train_size])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Baseline model
    start_base = time.time()
    base_model = SmallCNN().float()
    train_model(base_model, train_loader)
    base_train_time = time.time() - start_base

    # Masked model
    start_masked = time.time()
    gate = PixelMaskGate(H, W)
    model_masked = SmallCNN(gate).float()
    train_model(model_masked, train_loader, gate=gate)
    masked_train_time = time.time() - start_masked

    # Evaluation
    example_img, _ = test_ds[0]
    example_img = example_img.unsqueeze(0).float().requires_grad_()

    acc_base, f1_base = eval_metrics(base_model, test_loader)
    acc_masked, f1_masked = eval_metrics(model_masked, test_loader)


    base_infer = time_forward_pass(base_model, example_img)
    masked_infer = time_forward_pass(model_masked, example_img)
    saliency = Saliency(base_model)
    sal_map = saliency.attribute(example_img).squeeze().detach().cpu().numpy()
    with torch.no_grad():
        _ = model_masked(example_img)
        mask_map = gate.mask.squeeze().detach().cpu().numpy()

    sal_metrics = compute_interpretability_metrics(
        sal_map, region_A, region_B, region_distractor_1, region_distractor_2
    )
    mask_metrics = compute_interpretability_metrics(
        mask_map, region_A, region_B, region_distractor_1, region_distractor_2
    )

    return {
        "seed": seed,
        "acc_base": acc_base,
        "f1_base": f1_base,
        "acc_masked": acc_masked,
        "f1_masked": f1_masked,
        "train_time_base": base_train_time,
        "train_time_masked": masked_train_time,
        "infer_time_base": base_infer,
        "infer_time_masked": masked_infer,
        # Saliency (baseline model)
        "sal_iou_true": sal_metrics["iou_true"],
        "sal_iou_distractor": sal_metrics["iou_distractor"],
        "sal_mass_true": sal_metrics["mass_true"],
        "sal_mass_distractor": sal_metrics["mass_distractor"],
        # PixelMaskGate
        "mask_iou_true": mask_metrics["iou_true"],
        "mask_iou_distractor": mask_metrics["iou_distractor"],
        "mask_mass_true": mask_metrics["mass_true"],
        "mask_mass_distractor": mask_metrics["mass_distractor"],
    }



import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed


def main():
    seeds = list(range(100))
    results = []

    with ProcessPoolExecutor(max_workers=5) as executor:  # adjust to match your CPU cores
        futures = {executor.submit(run_single_simulation, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✅ Seed {seed} completed.")
            except Exception as e:
                print(f"❌ Seed {seed} failed: {e}")

    df = pd.DataFrame(results)
    df.to_csv("sim_outputs/simulation_results.csv", index=False)


if __name__ == "__main__":
    main()
