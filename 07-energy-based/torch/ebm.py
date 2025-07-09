import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

train_data = torchvision.datasets.FashionMNIST("./data", train=True, download=True)
test_data = torchvision.datasets.FashionMNIST("./data", train=False, download=True)


def preprocess(imgs):
    imgs = (imgs.astype("float32") - 127.5) / 127.5
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=-1.0)
    imgs = np.expand_dims(imgs, 1)
    return torch.FloatTensor(imgs)


x_train = preprocess(train_data.data.numpy())
x_test = preprocess(test_data.data.numpy())

train_dataset = TensorDataset(x_train)
test_dataset = TensorDataset(x_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class EBMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2 * 2 * 64, 64)
        self.ebm_output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        x = self.flatten(x)
        x = F.silu(self.dense(x))
        return self.ebm_output(x)


def generate_samples(model, inp_imgs, steps, step_size, noise):
    inp_imgs = inp_imgs.clone().detach().requires_grad_(True)
    for _ in range(steps):
        inp_imgs = inp_imgs + torch.randn_like(inp_imgs) * noise
        inp_imgs = torch.clamp(inp_imgs, -1.0, 1.0)
        if inp_imgs.grad is not None:
            inp_imgs.grad.zero_()
        out_score = -model(inp_imgs)
        out_score.backward(torch.ones_like(out_score))
        grads = inp_imgs.grad
        if grads is not None:
            grads = torch.clamp(grads, -0.03, 0.03)
            with torch.no_grad():
                inp_imgs = inp_imgs - step_size * grads
                inp_imgs = torch.clamp(inp_imgs, -1.0, 1.0)
        inp_imgs = inp_imgs.detach().requires_grad_(True)
    return inp_imgs.detach()


class Buffer:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.examples = [torch.rand((1, 1, 32, 32)) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(128, 0.05)
        rand_imgs = torch.rand((n_new, 1, 32, 32)) * 2 - 1
        if n_new < 128:
            old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)
            inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0)
        else:
            inp_imgs = rand_imgs
        inp_imgs = generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size, noise=noise
        )
        new_examples = torch.split(inp_imgs, 1, dim=0)
        self.examples = list(new_examples) + self.examples
        self.examples = self.examples[:8192]
        return inp_imgs


class EBM(nn.Module):
    def __init__(self):
        super(EBM, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2 * 2 * 64, 64)
        self.ebm_output = nn.Linear(64, 1)

        self.buffer = Buffer(self)
        self.alpha = 0.1

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        x = self.flatten(x)
        x = F.silu(self.dense(x))
        return self.ebm_output(x)

    def train_step(self, real_imgs):
        real_imgs = real_imgs + torch.randn_like(real_imgs) * 0.005
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(steps=60, step_size=10, noise=0.005)
        batch_size = real_imgs.shape[0]
        fake_imgs = fake_imgs[:batch_size]
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = torch.split(self(inp_imgs), batch_size, dim=0)
        cdiv_loss = torch.mean(fake_out, dim=0) - torch.mean(real_out, dim=0)
        reg_loss = self.alpha * torch.mean(real_out**2 + fake_out**2, dim=0)
        loss = reg_loss + cdiv_loss
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        with torch.no_grad():
            for param, grad in zip(self.parameters(), grads):
                param -= 0.0001 * grad
        return {
            "loss": loss,
            "reg": reg_loss,
            "cdiv": cdiv_loss,
            "real": torch.mean(real_out, dim=0),
            "fake": torch.mean(fake_out, dim=0),
        }

    def test_step(self, real_imgs):
        batch_size = real_imgs.shape[0]
        fake_imgs = torch.rand((batch_size, 1, 32, 32)) * 2 - 1
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = torch.split(self(inp_imgs), 2, dim=0)
        cdiv = torch.mean(fake_out, dim=0) - torch.mean(real_out, dim=0)
        return {
            "cdiv": cdiv,
            "real": torch.mean(real_out, dim=0),
            "fake": torch.mean(fake_out, dim=0),
        }


ebm = EBM()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"./logs/06-{timestamp}")

for epoch in range(60):
    for batch_idx, (real_imgs,) in enumerate(train_loader):
        metrics = ebm.train_step(real_imgs)
        step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("train/loss", metrics["loss"], step)
        writer.add_scalar("train/reg", metrics["reg"], step)
        writer.add_scalar("train/cdiv", metrics["cdiv"], step)
        writer.add_scalar("train/real", metrics["real"], step)
        writer.add_scalar("train/fake", metrics["fake"], step)

start_imgs = torch.rand((10, 1, 32, 32)) * 2 - 1
gen_img = generate_samples(
    ebm,
    start_imgs,
    steps=1000,
    step_size=10,
    noise=0.005,
)

torchvision.utils.save_image(gen_img, "generated_samples.png", normalize=True, nrow=5)

writer.close()
