import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

train_data = torchvision.datasets.FashionMNIST("./data", train=True, download=True)
test_data = torchvision.datasets.FashionMNIST("./data", train=False, download=True)


def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, 1)
    return torch.FloatTensor(imgs)


x_train = preprocess(train_data.data.numpy())
y_train = torch.LongTensor(train_data.targets)
x_test = preprocess(test_data.data.numpy())
y_test = torch.LongTensor(test_data.targets)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.shape_before_flattening = (128, 4, 4)
        self.fc = nn.Linear(128 * 4 * 4, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


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

        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)

        batch_size = x.size(0)
        x = x.view(batch_size, *self.shape_before_flattening)

        x = F.relu(self.conv_transpose1(x))
        x = F.relu(self.conv_transpose2(x))
        x = F.relu(self.conv_transpose3(x))

        x = torch.sigmoid(self.final_conv(x))

        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for epoch in tqdm(range(5)):
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

example_images = x_test[:5000]

model.eval()

with torch.no_grad():
    embeddings = model.encoder(example_images).numpy()

plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=3)
plt.show()

mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = torch.FloatTensor(np.random.uniform(mins, maxs, size=(18, 2)))

with torch.no_grad():
    reconstructions = model.decoder(sample).numpy()

plt.figure(figsize=(8, 8))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.imshow(reconstructions[i].squeeze(), cmap="gray")
    plt.axis("off")
plt.show()
