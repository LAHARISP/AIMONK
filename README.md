# Aimonk Multi-Label Image Classification

This project implements a **multi-label image classification model** using **PyTorch**.  
Each image may contain multiple attributes among four possible labels: Attr1, Attr2, Attr3, Attr4.

---

## 📌 Dataset

- Images stored in `images/`
- Labels provided in `labels.txt`
- Label values:
  - `1` → Attribute present
  - `0` → Attribute absent
  - `NA` → Attribute information unavailable

Images with `NA` values are **not ignored**.

---

## 🧠 Model

- Architecture: **ResNet18**
- Pretrained on **ImageNet**
- Final layer modified for **4 attributes**
- Activation: **Sigmoid**
- Loss: **Masked BCEWithLogitsLoss**

---

## ❓ Handling Missing Labels (NA)

`NA` labels are handled using a **masking strategy**:

- Valid labels → Included in loss
- NA labels → Ignored in loss computation

---

Output of training:

## 🚀 Training<img width="804" height="673" alt="Screenshot 2026-02-17 210000" src="https://github.com/user-attachments/assets/de9d9959-13bf-42ca-afcb-9322153dac73" />



<img width="698" height="369" alt="Screenshot 2026-02-17 210511" src="https://github.com/user-attachments/assets/82cd1f55-7e34-487d-86c2-6139de7c0d45" />


Run:
```bash
python train.py
python inference.py



