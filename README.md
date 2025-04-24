# Diabetic Retinopathy Diagnosis Using a Hybrid EfficientNet-ResNet Model with Coordinate Attention

This repository contains a deep learning model for diagnosing diabetic retinopathy from retinal fundus images.
It combines the strengths of **EfficientNet** and **ResNet** backbones along with **Coordinate Attention** to improve feature representation.

## 🧠 Model Architecture
- EfficientNetB0 + ResNet50 hybrid feature extractor.
- Coordinate Attention module to enhance spatial and channel-wise information.
- Fully connected classifier to output the DR grade.

## 📂 Dataset
We use the [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/datasets/mariaherrerot/aptos2019).

Organize your dataset folder as follows:

```
aptos2019/
├── train/
│   ├── 0/  # No DR
│   ├── 1/  # Mild
│   ├── 2/  # Moderate
│   ├── 3/  # Severe
│   └── 4/  # Proliferative DR
├── val/
│   ├── 0/
│   └── 1/
       ...
```

## 🚀 Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train.py
```

### Evaluate the model
```bash
python evaluate.py
```

## 📦 Files
- `train.py` – Training pipeline.
- `evaluate.py` – Evaluation script.
- `dataset.py - Custom PyTorch Dataset.
- `model.py` – Hybrid model with Coordinate Attention.
- `config.py` – Configurations for training.
- `requirements.txt` – Required libraries.

---

Created by Mahabube Alahi Atik for DR Diagnosis.
