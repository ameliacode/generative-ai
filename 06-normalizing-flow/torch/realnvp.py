from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class MoonDataset(Dataset):
    def __init__(self):
        data, labels = make_moons(3000, noise=0.05)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        mean = self.data.mean(dim=0)
        std = self.data.std(dim=0)
        self.data = (self.data - mean) / std

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class Coupling(nn.Module):
    def __init__(self):
        super(Coupling, self).__init__()
        self.s_layer_1 = nn.Linear(2, 256)
        self.s_layer_n = nn.Linear(256, 256)
        self.s_layer_5 = nn.Linear(256, 2)

        self.t_layer_1 = nn.Linear(2, 256)
        self.t_layer_n = nn.Linear(256, 256)
        self.t_layer_5 = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        s = self.relu(self.s_layer_1(x))
        s = self.relu(self.s_layer_n(s))
        s = self.relu(self.s_layer_n(s))
        s = self.relu(self.s_layer_n(s))
        s = self.tanh(self.s_layer_5(s))

        t = self.relu(self.t_layer_1(x))
        t = self.relu(self.t_layer_n(t))
        t = self.relu(self.t_layer_n(t))
        t = self.relu(self.t_layer_n(t))
        t = self.t_layer_5(t)

        return s, t


class RealNVP(nn.Module):
    def __init__(self, input_dim, coupling_layers, coupling_dim, regularization):
        super(RealNVP, self).__init__()
        self.coupling_layers = coupling_layers
        self.distribution = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
        self.register_buffer(
            "masks",
            torch.tensor(
                np.array([[0, 1], [1, 0]] * (coupling_layers // 2), dtype="float32")
            ),
        )
        self.layers_list = nn.ModuleList([Coupling() for i in range(coupling_layers)])

    def forward(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s = s * reversed_mask
            t = t * reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s))
                + x_masked
            )
            log_det_inv = log_det_inv + gate * torch.sum(s, dim=1)
        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -torch.mean(log_likelihood)

    def sample(self, n_samples):
        z = self.distribution.sample((n_samples,))
        x, _ = self(z, training=False)
        return x


dataset = MoonDataset()
model = RealNVP(input_dim=2, coupling_layers=6, coupling_dim=256, regularization=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"./logs/06-{timestamp}")

for epoch in range(100):  # 300
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = model.log_loss(data)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    writer.add_scalar("epoch_loss", avg_loss, epoch)

    if epoch % 10 == 0:
        with torch.no_grad():
            samples = model.sample(1000)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.scatter(dataset.data[:, 0], dataset.data[:, 1], alpha=0.6, s=1)
            ax1.set_title("Original Data")
            ax1.set_xlim(-3, 3)
            ax1.set_ylim(-3, 3)

            ax2.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
            ax2.set_title(f"Generated Samples - Epoch {epoch}")
            ax2.set_xlim(-3, 3)
            ax2.set_ylim(-3, 3)

            writer.add_figure(f"Samples/Epoch_{epoch}", fig, epoch)
            plt.close(fig)

writer.close()
