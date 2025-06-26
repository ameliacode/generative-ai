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
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    self.image_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image


class Critic(nn.Module):
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
        x = self.conv5(x)
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


class WGAN_GP:
    def __init__(self, device=None, critic_steps=5, gp_weight=10.0):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.generator = Generator().to(self.device)
        self.critic = Critic().to(self.device)  # Changed from discriminator
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

        # Initialize weights
        self._init_weights()

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            betas=(0.0, 0.9),  # WGAN-GP recommended settings
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=0.0001,
            betas=(0.0, 0.9),  # WGAN-GP recommended settings
        )

    def _init_weights(self):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.generator.apply(init_func)
        self.critic.apply(init_func)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)

        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)

        pred = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        norm = torch.sqrt(torch.sum(gradients**2, dim=1))

        gp = torch.mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images, step):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        for _ in range(self.critic_steps):
            self.c_optimizer.zero_grad()

            real_pred = self.critic(real_images)

            noise = torch.randn(batch_size, 100).to(self.device)
            fake_images = self.generator(noise).detach()
            fake_pred = self.critic(fake_images)

            c_wass_loss = torch.mean(fake_pred) - torch.mean(real_pred)

            c_gp = self.gradient_penalty(batch_size, real_images, fake_images)

            c_loss = c_wass_loss + self.gp_weight * c_gp

            c_loss.backward()
            self.c_optimizer.step()

        self.g_optimizer.zero_grad()

        noise = torch.randn(batch_size, 100).to(self.device)
        fake_images = self.generator(noise)
        fake_pred = self.critic(fake_images)

        g_loss = -torch.mean(fake_pred)

        g_loss.backward()
        self.g_optimizer.step()

        if self.writer:
            self.writer.add_scalar("epoch_c_loss", c_loss.item(), step)
            self.writer.add_scalar("epoch_g_loss", g_loss.item(), step)
            self.writer.add_scalar("epoch_c_wass_loss", c_wass_loss.item(), step)
            self.writer.add_scalar("epoch_c_gp", c_gp.item(), step)

        return c_loss.item(), g_loss.item(), c_wass_loss.item(), c_gp.item()

    def generate(self, num_samples=64):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, 100).to(self.device)
            fake_images = self.generator(noise)
            fake_images = (fake_images + 1) / 2  # Normalize to [0,1]
        self.generator.train()
        return fake_images

    def save(self, path):
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "g_optimizer": self.g_optimizer.state_dict(),
                "c_optimizer": self.c_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint["generator"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "g_optimizer" in checkpoint:
            self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        if "c_optimizer" in checkpoint:
            self.c_optimizer.load_state_dict(checkpoint["c_optimizer"])


def train(data_path, epochs=100, batch_size=128, save_every=20):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = CelebADataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    wgan_gp = WGAN_GP(critic_steps=5, gp_weight=10.0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/04-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_dir = f"./logs/04-{timestamp}"
    wgan_gp.writer = SummaryWriter(log_dir)

    step = 0
    for epoch in tqdm(range(epochs), desc="Training"):
        c_losses = []
        g_losses = []
        wass_losses = []
        gp_losses = []

        for batch_idx, real_images in enumerate(dataloader):
            c_loss, g_loss, wass_loss, gp_loss = wgan_gp.train_step(real_images, step)
            c_losses.append(c_loss)
            g_losses.append(g_loss)
            wass_losses.append(wass_loss)
            gp_losses.append(gp_loss)
            step += 1

        if (epoch + 1) % save_every == 0:
            samples = wgan_gp.generate(64)
            grid = vutils.make_grid(samples, nrow=8, padding=2)
            vutils.save_image(grid, f"{output_dir}/epoch_{epoch+1}.png")

            wgan_gp.writer.add_image("Generated_Images", grid, epoch)

            wgan_gp.save(f"{output_dir}/checkpoint_epoch_{epoch+1}.pth")

    final_samples = wgan_gp.generate(100)
    final_grid = vutils.make_grid(final_samples, nrow=10, padding=2)
    vutils.save_image(final_grid, f"{output_dir}/final_results.png")
    wgan_gp.save(f"{output_dir}/wgan-gp_final.pth")

    wgan_gp.writer.close()

    return wgan_gp, output_dir


if __name__ == "__main__":
    data_path = "./data/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"
    wgan_gp, output_dir = train(data_path, epochs=100)
