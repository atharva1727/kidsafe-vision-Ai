# 🧠 KidSafe Vision AI – Intelligent Child Safety Image Classifier

<p align="center">
<img src="Images/model-workflow.png" width="1000"/>
</p>

<h1 align="center">KidSafe Vision AI</h1>

<h3 align="center">
🚀 AI-Powered Content Safety System for Children
</h3>

<p align="center">
Protecting children from harmful visual content using Machine Learning and Computer Vision
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10-blue"/>
<img src="https://img.shields.io/badge/Flask-WebApp-success"/>
<img src="https://img.shields.io/badge/AI-MachineLearning-orange"/>
<img src="https://img.shields.io/badge/ComputerVision-ImageClassification-purple"/>
<img src="https://img.shields.io/badge/Status-Completed-brightgreen"/>
<img src="https://img.shields.io/badge/Accuracy-94.3%25-red"/>

</p>

---

# 📌 Overview

KidSafe Vision AI is an intelligent image classification system designed to create a safer digital environment for children.

Children frequently encounter harmful or inappropriate content while browsing websites, social media platforms, or mobile applications. Traditional filtering approaches often struggle to identify visual content accurately and at scale.

KidSafe Vision uses Machine Learning and Computer Vision techniques to automatically analyze images and classify them as:

✅ SAFE

❌ UNSAFE

The system instantly provides prediction results with confidence scores, enabling real-time content moderation and child protection.

---

# 🌟 Problem Statement

Children frequently encounter:

- Violent imagery
- Disturbing visual content
- Harmful online media
- Inappropriate digital material

Current systems often require manual moderation which is:

❌ Slow

❌ Expensive

❌ Difficult to scale

KidSafe Vision provides an automated AI-powered solution.

---

# 🎯 Objectives

✔ Build a child safety image classifier

✔ Binary classification (Safe / Unsafe)

✔ Real-time image prediction

✔ High accuracy with minimal false positives

✔ Interactive web interface

✔ Fast CPU-based deployment

---

# ⚡ Technology Stack

## Programming Language

- Python 3.10

## Framework

- Flask

## Machine Learning Models

- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

## AI Domain

- Machine Learning
- Computer Vision
- Image Classification

## Deployment

- Render

---

# 🔥 Core Features

### 🧠 Intelligent Image Classification

Analyzes uploaded images and determines whether content is safe or unsafe.

---

### ⚡ Real-Time Prediction

Fast inference with response time below 200 milliseconds.

---

### 📊 Confidence Scoring

Returns prediction probabilities for better transparency.

---

### 🖥 Interactive Web Interface

Simple and user-friendly interface for testing images.

---

### 🚀 Lightweight Deployment

Optimized to run efficiently on CPU without requiring GPU resources.

---

# 📷 Project Preview

## 🧠 AI Workflow Architecture

![Workflow](https://github.com/atharva1727/kidsafe-vision-Ai/blob/main/AI%20(2).jpg)

---

## ⚡ Prediction Interface

![Interface](https://github.com/atharva1727/kidsafe-vision-Ai/blob/main/AI%20(3).jpg)

---

## 📊 Model Performance Dashboard

![Performance](https://github.com/atharva1727/kidsafe-vision-Ai/blob/main/AI.jpg)

---

# 🏗 System Workflow

User Uploads Image

⬇

Image Preprocessing

⬇

Feature Extraction

⬇

ML Model Prediction

⬇

Confidence Score Generation

⬇

SAFE / UNSAFE Result

---

# 📈 Model Performance

| Metric | Score |
|----------|---------|
| Accuracy | 94.3% |
| Precision | 0.93 |
| Recall | 0.91 |
| F1 Score | 0.93 |
| False Positive Rate | 5.7% |
| Inference Speed | ~150ms |

---

# 🌍 Potential Applications

### 👨‍👩‍👧 Parental Control Systems

Filter unsafe content before children access it.

### 🏫 Educational Platforms

Provide secure learning environments.

### 📱 Social Media Platforms

Automate content moderation.

### 🌐 Browser Extensions

Enable real-time web filtering.

### 🛡 Government & NGOs

Support child online safety initiatives.

---

# 🚀 Future Enhancements

### Multi-Class Classification

Detect:

- Violence
- Hate symbols
- Drug-related content
- Adult content

### Video Analysis

Frame-by-frame live detection.

### Browser Extension

Real-time webpage filtering.

### Federated Learning

Improve privacy without sharing user data.

### Multi-modal AI

Combine image and text understanding.

---

# 👨‍💻 Author

**Atharv Shevate**

AI & Full Stack Developer | Machine Learning Enthusiast | Computer Engineer

---

<p align="center">

⭐ If you found this project useful, consider giving it a star ⭐

</p>




# Child Safety Image Classifier

> **Model Used:** EfficientNetB0 (Transfer Learning from ImageNet)  
> **Expected Accuracy:** 90%+ with a balanced dataset of 500+ images per class

---


## Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 (pretrained on ImageNet, top layers fine-tuned)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, ReLU) → Dropout(0.4)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(1, Sigmoid)  →  0 = SAFE  |  1 = UNSAFE
```


### Why EfficientNetB0?
- Pretrained on 1.2M ImageNet images — excellent visual feature extraction
- Compact yet powerful (5.3M params)
- Achieves high accuracy with small datasets via transfer learning
- Scales efficiently (no need for a GPU farm)

---


## Dataset Setup

Place your images in the following structure **before training**:

```
dataset/
├── safe/          ← child-appropriate images
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
└── unsafe/        ← images NOT appropriate for children
    ├── img1.jpg
    ├── img2.png
    └── ...
```

**Recommendations:**
- Minimum: 300 images per class (more = better)
- Recommended: 1000+ images per class
- Balanced classes give the best results
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

Training runs in **2 phases**:
1. **Phase 1** — Only the custom classification head is trained (fast convergence)
2. **Phase 2** — Top 30 layers of EfficientNetB0 are unfrozen and fine-tuned (higher accuracy)

Outputs saved to `models/`:
- `child_safety_model.h5` — best model checkpoint
- `training_curve_phase1.png` / `training_curve_phase2.png`
- `confusion_matrix.png`
- `training_history.json`

---

## Predict a Single Image

```bash
python predict.py path/to/image.jpg
```

Example output:
```
=============================================
  Image   : photo.jpg
  Result  : ✅ SAFE
  Confidence: 97.32%
  Raw score : 0.0268  (>0.5 = unsafe)
=============================================
```

---

## Evaluate on Test Set

```bash
python evaluate.py --test_dir dataset/test --model models/child_safety_model.h5
```

Generates a full classification report + ROC-AUC score.

---

## Configuration

Edit `CONFIG` in `train.py` to change:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `epochs` | 30 | Max epochs (early stopping applies) |
| `learning_rate` | 1e-4 | Phase 1 LR |
| `fine_tune_lr` | 1e-5 | Phase 2 fine-tuning LR |
| `validation_split` | 0.2 | 20% for validation |

---

## Project Structure

```
child_safety_classifier/
├── dataset/
│   ├── safe/          ← ADD YOUR IMAGES HERE
│   └── unsafe/        ← ADD YOUR IMAGES HERE
├── models/            ← auto-created during training
├── train.py           ← main training script
├── predict.py         ← single image inference
├── evaluate.py        ← batch evaluation on test set
├── requirements.txt
└── README.md
```
