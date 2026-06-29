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
