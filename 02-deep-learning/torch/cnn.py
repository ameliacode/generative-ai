import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True)
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True)

x_train = torch.FloatTensor(train_data.data).permute(0, 3, 1, 2) / 255.0  # (N, C, H, W)
y_train = torch.LongTensor(train_data.targets)
x_test = torch.FloatTensor(test_data.data).permute(0, 3, 1, 2) / 255.0
y_test = torch.LongTensor(test_data.targets)

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(
            64 * 8 * 8, 128
        )  # 32x32 -> 16x16 -> 8x8 after 2 stride-2 convs
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn5(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm(range(10)):
    for i in range(0, len(x_train), 32):
        batch_x = x_train[i : i + 32]
        batch_y = y_train[i : i + 32]

        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    test_pred = model(x_test)
    accuracy = (test_pred.argmax(1) == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.3f}")

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

preds = test_pred.argmax(1).numpy()
indices = np.random.choice(len(x_test), 10)

fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i, idx in enumerate(indices):
    img = test_data.data[idx]
    axes[i].imshow(img)
    axes[i].set_title(f"Pred: {CLASSES[preds[idx]]}\nActual: {CLASSES[y_test[idx]]}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()
