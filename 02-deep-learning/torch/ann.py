import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True)
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True)

x_train = torch.FloatTensor(train_data.data) / 255.0
y_train = torch.LongTensor(train_data.targets)
x_test = torch.FloatTensor(test_data.data) / 255.0
y_test = torch.LongTensor(test_data.targets)

x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)

model = nn.Sequential(
    nn.Linear(32 * 32 * 3, 200),
    nn.ReLU(),
    nn.Linear(200, 120),
    nn.ReLU(),
    nn.Linear(120, 10),
)

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

    print(f"Epoch {epoch+1}/10 completed")

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
