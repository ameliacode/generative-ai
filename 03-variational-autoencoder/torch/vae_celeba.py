import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

train_dataset = CelebADataset(
    root_dir="./data/celeba-dataset/versions/2/img_align_celeba/img_align_celeba",
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.shape_before_flattening = (128, 4, 4)
        self.fc_mu = nn.Linear(128 * 4 * 4, 2)
        self.fc_logvar = nn.Linear(128 * 4 * 4, 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        z_mean = self.fc_mu(x)
        z_log_var = self.fc_logvar(x)

        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, shape_before_flattening=(128, 4, 4)):
        super(Decoder, self).__init__()

        self.shape_before_flattening = shape_before_flattening
        self.fc = nn.Linear(2, np.prod(shape_before_flattening))

        self.conv_transpose1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv_transpose2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv_transpose3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)

        x = F.leaky_relu(self.conv_transpose1(x))
        x = F.leaky_relu(self.conv_transpose2(x))
        x = F.leaky_relu(self.conv_transpose3(x))
        x = torch.sigmoid(self.final_conv(x))

        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def sampling(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        recon_x = self.decoder(z)
        return z_mean, z_log_var, recon_x


def loss_fn(x, z_mean, z_log_var, recon_x):
    reconstruction_loss = torch.mean(
        500 * F.binary_cross_entropy(recon_x, x, reduction="none").sum(dim=(1, 2, 3))
    )
    kl_loss = torch.mean(
        torch.sum(-0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()), dim=1)
    )

    return reconstruction_loss + kl_loss


model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in tqdm(range(5)):
    for x in train_loader:
        z_mean, z_log_var, recon_x = model(x)
        loss = loss_fn(x, z_mean, z_log_var, recon_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

grid_width, grid_height = (10, 3)
sample = torch.FloatTensor(np.random.normal(size=(grid_width * grid_height, 2)))

with torch.no_grad():
    reconstructions = model.decoder(sample)

reconstructions = reconstructions.permute(0, 2, 3, 1).cpu().numpy()

plt.figure(figsize=(18, 5))
for i in range(grid_width * grid_height):
    plt.subplot(grid_height, grid_width, i + 1)
    plt.imshow(np.clip(reconstructions[i], 0, 1))
    plt.axis("off")
plt.show()
