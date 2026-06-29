"""
Predict whether an image is safe or unsafe for children.
Usage: python predict.py <image_path> [--model models/child_safety_model.h5]
"""

import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = (224, 224)
CLASS_NAMES = {0: "SAFE", 1: "UNSAFE"}
CLASS_EMOJI = {0: "✅", 1: "🚫"}


def load_model(model_path="models/child_safety_model.h5"):
    print(f"Loading model from {model_path} ...")
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image_path):
    img = preprocess_image(image_path)
    prob = model.predict(img, verbose=0)[0][0]
    # prob ~ 0 => safe, prob ~ 1 => unsafe
    label_idx = int(prob > 0.5)
    label = CLASS_NAMES[label_idx]
    confidence = prob if label_idx == 1 else 1 - prob

    print("\n" + "=" * 45)
    print(f"  Image   : {image_path}")
    print(f"  Result  : {CLASS_EMOJI[label_idx]} {label}")
    print(f"  Confidence: {confidence * 100:.2f}%")
    print(f"  Raw score : {prob:.4f}  (>0.5 = unsafe)")
    print("=" * 45 + "\n")
    return label, float(confidence)


def main():
    parser = argparse.ArgumentParser(description="Child Safety Image Predictor")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="models/child_safety_model.h5", help="Path to trained model")
    args = parser.parse_args()

    model = load_model(args.model)
    predict(model, args.image)


if __name__ == "__main__":
    main()
