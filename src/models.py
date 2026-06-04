"""Shared CNN architectures used across more than one experiment.

Only models that are imported by multiple scripts live here. Experiment-specific
architectures (the various ``SimpleCNN`` variants, ``DeeperCNN``, ``SingleResCNN``,
``SimpleCNN3D``) remain defined inline in their own scripts under ``experiments/``,
because they differ subtly from one another.
"""

import torch
import torch.nn as nn


class MultiResCNN(nn.Module):
    """Two-branch CNN that ingests a full-resolution and a down-sampled field.

    Each branch has its own conv stack; the flattened features are concatenated
    before the classifier head. The first fully connected layer is sized lazily
    on the first forward pass so the model adapts to whatever spatial dimensions
    the two resolutions produce.

    Imported by ``experiments/mixed_resolution.py`` (training) and
    ``saliency/multires_saliency.py`` (post-hoc saliency).
    """

    def __init__(self, input_channels_1=1, input_channels_k=1):
        super().__init__()
        # Convolutional layers for resolution 1 (full resolution).
        self.conv1_res1 = nn.Conv2d(input_channels_1, 8, kernel_size=3, padding=1)
        self.conv2_res1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool_res1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layers for resolution k (down-sampled).
        self.conv1_resk = nn.Conv2d(input_channels_k, 8, kernel_size=3, padding=1)
        self.conv2_resk = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool_resk = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (fc1 in_features adjusted dynamically in forward).
        self.fc1 = nn.Linear(1, 128)  # Placeholder size; resized on first forward.
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_res1, x_resk):
        # Process resolution 1 independently.
        x1 = self.pool_res1(torch.relu(self.conv1_res1(x_res1)))
        x1 = self.pool_res1(torch.relu(self.conv2_res1(x1)))
        x1 = x1.reshape(x1.size(0), -1)  # Flatten

        # Process resolution k independently.
        xk = self.pool_resk(torch.relu(self.conv1_resk(x_resk)))
        xk = self.pool_resk(torch.relu(self.conv2_resk(xk)))
        xk = xk.reshape(xk.size(0), -1)  # Flatten

        # Concatenate flattened outputs.
        x = torch.cat([x1, xk], dim=1)

        # Lazily size the first fully connected layer if needed.
        if x.size(1) != self.fc1.in_features:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
