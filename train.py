import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Config
# -----------------------------
IMAGE_DIR = r"C:\Users\LAHARI\Desktop\AIMONK\images-20260217T145008Z-1-001\images"
LABEL_FILE = "labels.txt"
NUM_ATTRS = 4
BATCH_SIZE = 8          # Safe for CPU / small GPU
EPOCHS = 20             # Better for ~974 images
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATTR_NAMES = ["Attr1", "Attr2", "Attr3", "Attr4"]

# -----------------------------
# Label Parsing
# -----------------------------
def load_labels(label_file):
    labels = {}

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            img_name = parts[0]
            attrs = parts[1:]

            label_vector = []
            mask_vector = []

            for attr in attrs:
                if attr == "NA":
                    label_vector.append(0)   # dummy
                    mask_vector.append(0)    # ignore
                else:
                    label_vector.append(int(attr))
                    mask_vector.append(1)

            labels[img_name] = (
                np.array(label_vector, dtype=np.float32),
                np.array(mask_vector, dtype=np.float32)
            )

    return labels

# -----------------------------
# Dataset
# -----------------------------
class MultiLabelDataset(Dataset):
    def __init__(self, image_dir, labels_dict, transform=None):
        self.image_dir = image_dir
        self.labels_dict = labels_dict
        self.transform = transform

        self.image_names = [
            img_name for img_name in labels_dict.keys()
            if os.path.exists(os.path.join(image_dir, img_name))
        ]

        missing = [
            img_name for img_name in labels_dict.keys()
            if not os.path.exists(os.path.join(image_dir, img_name))
        ]

        print(f"⚠️ Missing images: {len(missing)}")
        print(f"✅ Using images: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        labels, mask = self.labels_dict[img_name]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels), torch.tensor(mask)

# -----------------------------
# Transforms (ImageNet style)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Data
# -----------------------------
labels_dict = load_labels(LABEL_FILE)

print(f"Total labels in file: {len(labels_dict)}")
print(f"Images in folder: {len(os.listdir(IMAGE_DIR))}")

dataset = MultiLabelDataset(IMAGE_DIR, labels_dict, transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,    # Windows safe
)

# -----------------------------
# Model (Pretrained ResNet18)
# -----------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_ATTRS)
model = model.to(DEVICE)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Loop
# -----------------------------
loss_history = []
iteration = 0

model.train()

for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

    for images, labels, mask in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss_matrix = criterion(outputs, labels)
        masked_loss = loss_matrix * mask

        if mask.sum() == 0:
            continue

        loss = masked_loss.sum() / mask.sum()

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        iteration += 1

        print(f"Iter {iteration} | Loss: {loss.item():.4f}")

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "model.pth")
print("\n✅ Model saved as model.pth")

# -----------------------------
# Plot Loss Curve
# -----------------------------
plt.plot(loss_history)
plt.xlabel("iteration_number")
plt.ylabel("training_loss")
plt.title("Aimonk_multilabel_problem")
plt.show()
