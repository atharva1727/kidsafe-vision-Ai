"""
Evaluate the trained model on a test directory.
Usage: python evaluate.py --test_dir dataset/test  (or any dir with safe/unsafe sub-folders)
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate(model_path, test_dir, batch_size=32, image_size=(224, 224)):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded: {model_path}")

    datagen = ImageDataGenerator(rescale=1.0/255)
    test_gen = datagen.flow_from_directory(
        test_dir, target_size=image_size,
        batch_size=batch_size, class_mode="binary",
        shuffle=False,
    )

    probs = model.predict(test_gen, verbose=1).flatten()
    preds = (probs > 0.5).astype(int)
    true  = test_gen.classes
    classes = list(test_gen.class_indices.keys())

    print("\n=== Evaluation Results ===")
    print(classification_report(true, preds, target_names=classes))
    print(f"ROC-AUC: {roc_auc_score(true, probs):.4f}")

    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Test Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("models/test_confusion_matrix.png", dpi=150)
    print("Saved: models/test_confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/child_safety_model.h5")
    parser.add_argument("--test_dir", default="dataset/test")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.model, args.test_dir, args.batch_size)
