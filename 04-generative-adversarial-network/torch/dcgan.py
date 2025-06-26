import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    self.image_files.append(os.path.join(root, f))

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_files)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1, 1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.conv5 = nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        return x


class DCGAN:
    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.criterion = nn.BCELoss()

        self.writer = None

    def train_step(self, real_images, step):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        self.d_optimizer.zero_grad()

        real_pred = self.discriminator(real_images)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        d_real_loss = self.criterion(real_pred, real_labels)

        noise = torch.randn(batch_size, 100).to(self.device)
        fake_images = self.generator(noise)
        fake_pred = self.discriminator(fake_images.detach())
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        d_fake_loss = self.criterion(fake_pred, fake_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()

        self.g_optimizer.zero_grad()
        fake_pred = self.discriminator(fake_images)
        g_loss = self.criterion(fake_pred, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        if self.writer:
            self.writer.add_scalar("epoch_d_loss", d_loss.item(), step)
            self.writer.add_scalar("epoch_g_loss", g_loss.item(), step)

            d_real_acc = (real_pred > 0.5).float().mean().item()
            d_fake_acc = (fake_pred < 0.5).float().mean().item()
            d_acc = (d_real_acc + d_fake_acc) / 2

            g_acc = (fake_pred > 0.5).float().mean().item()

            self.writer.add_scalar("epoch_d_acc", d_acc, step)
            self.writer.add_scalar("epoch_g_acc", g_acc, step)

        return d_loss.item(), g_loss.item()

    def generate(self, num_samples=64):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, 100).to(self.device)
            fake_images = self.generator(noise)
            fake_images = (fake_images + 1) / 2
        self.generator.train()
        return fake_images

    def save(self, path):
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "g_optimizer": self.g_optimizer.state_dict(),
                "d_optimizer": self.d_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        if "g_optimizer" in checkpoint:
            self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        if "d_optimizer" in checkpoint:
            self.d_optimizer.load_state_dict(checkpoint["d_optimizer"])


def train(data_path, epochs=100, batch_size=128, save_every=10):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = LegoDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dcgan = DCGAN()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/04-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_dir = f"logs/04-{timestamp}"
    dcgan.writer = SummaryWriter(log_dir)

    step = 0
    for epoch in tqdm(range(epochs)):
        d_losses = []
        g_losses = []

        for batch_idx, real_images in enumerate(dataloader):
            d_loss, g_loss = dcgan.train_step(real_images, step)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            step += 1

        if (epoch + 1) % save_every == 0:
            samples = dcgan.generate(64)
            grid = vutils.make_grid(samples, nrow=8, padding=2)
            vutils.save_image(grid, f"{output_dir}/epoch_{epoch+1}.png")

            dcgan.writer.add_image("Generated_Images", grid, epoch)

            dcgan.save(f"{output_dir}/checkpoint_epoch_{epoch+1}.pth")

    final_samples = dcgan.generate(100)
    final_grid = vutils.make_grid(final_samples, nrow=10, padding=2)
    vutils.save_image(final_grid, f"{output_dir}/final_results.png")
    dcgan.save(f"{output_dir}/dcgan_final.pth")

    dcgan.writer.close()

    return dcgan, output_dir


if __name__ == "__main__":
    data_path = "./data/lego-brick-images/versions/4/dataset/"
    dcgan, output_dir = train(data_path, epochs=100)
