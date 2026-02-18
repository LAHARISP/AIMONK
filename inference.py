import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "model.pth"
NUM_ATTRS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATTR_NAMES = ["Attr1", "Attr2", "Attr3", "Attr4"]

# -----------------------------
# Transform (ImageNet style)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet18(weights=None)  # Modern syntax
model.fc = nn.Linear(model.fc.in_features, NUM_ATTRS)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image_path, threshold=0.5):

    # ---- Safety checks ----
    if not image_path.strip():
        print("❌ No path entered.")
        return

    if not os.path.exists(image_path):
        print("❌ File does NOT exist.")
        return

    if os.path.isdir(image_path):
        print("❌ You entered a folder, not an image.")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)

    probs = probs.cpu().numpy()[0]

    # ---- Print probabilities ----
    print("\n📊 Prediction Probabilities:")
    for i, p in enumerate(probs):
        print(f"{ATTR_NAMES[i]}: {p:.3f}")

    # ---- Thresholding ----
    predicted_attrs = [
        ATTR_NAMES[i]
        for i, p in enumerate(probs)
        if p > threshold
    ]

    # ---- Final Output ----
    print("\n🔮 Predicted Attributes:")
    if predicted_attrs:
        for attr in predicted_attrs:
            print(f"✔ {attr}")
    else:
        print("No attributes detected")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    img_path = input("\nEnter image path: ")
    predict(img_path)
