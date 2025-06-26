import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

LABEL = "Smiling"
CLASSES = 2


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []

        for root, dirs, files in os.walk(
            os.path.join(root_dir, "img_align_celeba/img_align_celeba")
        ):
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    self.image_files.append(os.path.join(root, f))

        self.attributes = pd.read_csv(os.path.join(root_dir, "list_attr_celeba.csv"))
        self.labels = self.attributes[LABEL].tolist()
        self.int_labels = [1 if x == 1 else 0 for x in self.labels]
        self.one_hot_labels = torch.eye(CLASSES)[[self.int_labels]]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.one_hot_labels[idx]


def preprocess(x):
    return x


IMAGE_SIZE = 64
CHANNELS = 3


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            CHANNELS + CLASSES, 64, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0)

        self.dropout = nn.Dropout(0.3)

    def forward(self, critic_input, label_input):
        label_expanded = (
            label_input.unsqueeze(2).unsqueeze(3).expand(-1, -1, IMAGE_SIZE, IMAGE_SIZE)
        )
        x = torch.cat([critic_input, label_expanded], dim=1)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.dropout(x)
        x = self.conv5(x)
        x = x.flatten(start_dim=1)
        return x


Z_DIM = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            Z_DIM + CLASSES, 128, kernel_size=4, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.conv3 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.conv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.conv5 = nn.ConvTranspose2d(
            64, CHANNELS, kernel_size=4, stride=2, padding=1
        )

        self.bn1 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.1)

    def forward(self, generator_input, label_input):
        x = torch.cat([generator_input, label_input], dim=1)
        x = x.view(-1, Z_DIM + CLASSES, 1, 1)

        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.tanh(self.conv5(x))
        return x


class CGAN:
    def __init__(self, device=None, critic_steps=5, gp_weight=10.0):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.generator = Generator().to(self.device)
        self.critic = Critic().to(self.device)
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

        self._init_weights()

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            betas=(0.0, 0.9),
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=0.0001,
            betas=(0.0, 0.9),
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

    def gradient_penalty(
        self, batch_size, real_images, fake_images, image_one_hot_labels
    ):
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)

        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)

        pred = self.critic(interpolated, image_one_hot_labels)

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

    def train_step(self, data, step):
        real_images, one_hot_labels = data

        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        one_hot_labels = one_hot_labels.to(self.device)

        for i in range(self.critic_steps):
            random_latent_vectors = torch.randn(batch_size, Z_DIM).to(self.device)

            self.c_optimizer.zero_grad()

            fake_images = self.generator(random_latent_vectors, one_hot_labels)

            # Pass the original one_hot_labels to critic, let it handle expansion
            fake_predictions = self.critic(fake_images, one_hot_labels)
            real_predictions = self.critic(real_images, one_hot_labels)

            c_wass_loss = torch.mean(fake_predictions) - torch.mean(real_predictions)
            c_gp = self.gradient_penalty(
                batch_size, real_images, fake_images, one_hot_labels
            )
            c_loss = c_wass_loss + c_gp * self.gp_weight

            c_loss.backward()
            self.c_optimizer.step()

        random_latent_vectors = torch.randn(batch_size, Z_DIM).to(self.device)

        self.g_optimizer.zero_grad()

        fake_images = self.generator(random_latent_vectors, one_hot_labels)
        fake_predictions = self.critic(fake_images, one_hot_labels)
        g_loss = -torch.mean(fake_predictions)

        g_loss.backward()
        self.g_optimizer.step()

        if self.writer:
            self.writer.add_scalar("Loss/Critic", c_loss.item(), step)
            self.writer.add_scalar("Loss/Generator", g_loss.item(), step)
            self.writer.add_scalar("Loss/Wasserstein", c_wass_loss.item(), step)
            self.writer.add_scalar("Loss/GradientPenalty", c_gp.item(), step)

        return c_loss.item(), g_loss.item(), c_wass_loss.item(), c_gp.item()

    def generate_by_class(self, num_samples=10, class_label=0):
        self.generator.eval()
        with torch.no_grad():
            z_sample = torch.randn(num_samples, Z_DIM).to(self.device)
            if class_label == 0:
                class_labels = (
                    torch.tensor([[1, 0]])
                    .repeat(num_samples, 1)
                    .float()
                    .to(self.device)
                )
            else:
                class_labels = (
                    torch.tensor([[0, 1]])
                    .repeat(num_samples, 1)
                    .float()
                    .to(self.device)
                )
            fake_images = self.generator(z_sample, class_labels)
            fake_images = (fake_images + 1) / 2
        self.generator.train()
        return fake_images

    def generate(self, num_samples=64, labels=None):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, Z_DIM).to(self.device)
            if labels is None:
                labels = torch.eye(CLASSES)[
                    torch.randint(0, CLASSES, (num_samples,))
                ].to(self.device)
            fake_images = self.generator(noise, labels)
            fake_images = (fake_images + 1) / 2
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
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CelebADataset(data_path, transform=transform)

    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train = train_data

    cgan = CGAN(critic_steps=5, gp_weight=10.0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/04-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_dir = f"{output_dir}/logs"
    cgan.writer = SummaryWriter(log_dir)

    step = 0
    for epoch in tqdm(range(epochs), desc="Training"):
        c_losses = []
        g_losses = []
        wass_losses = []
        gp_losses = []

        for batch_idx, data in enumerate(train):
            c_loss, g_loss, wass_loss, gp_loss = cgan.train_step(data, step)
            c_losses.append(c_loss)
            g_losses.append(g_loss)
            wass_losses.append(wass_loss)
            gp_losses.append(gp_loss)
            step += 1

        if (epoch + 1) % save_every == 0:
            samples = cgan.generate(64)
            grid = vutils.make_grid(samples, nrow=8, padding=2)
            vutils.save_image(grid, f"{output_dir}/epoch_{epoch+1}.png")

            cgan.writer.add_image("Generated_Images", grid, epoch)

            cgan.save(f"{output_dir}/checkpoint_epoch_{epoch+1}.pth")

    final_samples = cgan.generate(100)
    final_grid = vutils.make_grid(final_samples, nrow=10, padding=2)
    vutils.save_image(final_grid, f"{output_dir}/final_results.png")
    cgan.save(f"{output_dir}/cgan_final.pth")

    cgan.writer.close()

    return cgan, output_dir


if __name__ == "__main__":
    data_path = "./data/celeba-dataset/versions/2/"
    cgan, output_dir = train(data_path, epochs=100)

    imgs_class_0 = cgan.generate_by_class(num_samples=10, class_label=0)
    grid_0 = vutils.make_grid(imgs_class_0, nrow=5, padding=2)
    vutils.save_image(grid_0, f"{output_dir}/class_0_samples.png")

    imgs_class_1 = cgan.generate_by_class(num_samples=10, class_label=1)
    grid_1 = vutils.make_grid(imgs_class_1, nrow=5, padding=2)
    vutils.save_image(grid_1, f"{output_dir}/class_1_samples.png")
