"""
Child Safety Image Classifier
Model: EfficientNetB0 (Transfer Learning)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

CONFIG = {
    "dataset_dir": "dataset",
    "image_size": (224, 224),
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 1e-4,
    "fine_tune_lr": 1e-5,
    "validation_split": 0.2,
    "model_save_path": "models/child_safety_model.h5",
    "history_save_path": "models/training_history.json",
    "classes": ["safe", "unsafe"],
    "seed": 42,
}

tf.random.set_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=CONFIG["validation_split"],
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=CONFIG["validation_split"])

    train_gen = train_datagen.flow_from_directory(
        CONFIG["dataset_dir"], target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"], class_mode="binary",
        subset="training", seed=CONFIG["seed"], shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        CONFIG["dataset_dir"], target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"], class_mode="binary",
        subset="validation", seed=CONFIG["seed"], shuffle=False,
    )

    print(f"\n Class mapping: {train_gen.class_indices}")
    print(f"  Training samples  : {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}\n")
    return train_gen, val_gen


def build_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*CONFIG["image_size"], 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid", name="child_safety_output")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    print(f"\nModel: EfficientNetB0 + Custom Head")
    print(f"  Total params: {model.count_params():,}\n")
    return model, base_model


def get_callbacks(phase="phase1"):
    return [
        ModelCheckpoint(CONFIG["model_save_path"], monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=f"models/logs/{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    ]


def plot_history(history, phase="phase1"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title(f"Accuracy ({phase})"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"Loss ({phase})"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"models/training_curve_{phase}.png", dpi=150)
    print(f"Saved: models/training_curve_{phase}.png")
    plt.close()


def plot_confusion_matrix(model, val_gen):
    val_gen.reset()
    preds = (model.predict(val_gen, verbose=1) > 0.5).astype(int).flatten()
    true  = val_gen.classes
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CONFIG["classes"], yticklabels=CONFIG["classes"])
    plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    print("Saved: models/confusion_matrix.png")
    plt.close()
    print("\nClassification Report:")
    print(classification_report(true, preds, target_names=CONFIG["classes"]))


def fine_tune(model, base_model, train_gen, val_gen):
    print("\nFine-tuning top 30 layers of EfficientNetB0...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["fine_tune_lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model.fit(train_gen, epochs=15, validation_data=val_gen, callbacks=get_callbacks("phase2"))


def main():
    print("=" * 60)
    print("  Child Safety Image Classifier")
    print("  Model: EfficientNetB0 (Transfer Learning)")
    print("=" * 60)
    for cls in CONFIG["classes"]:
        path = os.path.join(CONFIG["dataset_dir"], cls)
        count = len([f for f in os.listdir(path) if not f.startswith('.')]) if os.path.exists(path) else 0
        print(f"  {cls}: {count} images")

    os.makedirs("models/logs", exist_ok=True)
    train_gen, val_gen = build_data_generators()
    model, base_model = build_model()

    print("\nPhase 1: Training classification head...")
    history1 = model.fit(train_gen, epochs=CONFIG["epochs"], validation_data=val_gen, callbacks=get_callbacks("phase1"))
    plot_history(history1, "phase1")

    history2 = fine_tune(model, base_model, train_gen, val_gen)
    plot_history(history2, "phase2")

    print("\nFinal Evaluation:")
    results = model.evaluate(val_gen, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_confusion_matrix(model, val_gen)

    all_history = {
        "phase1": {k: [float(x) for x in v] for k, v in history1.history.items()},
        "phase2": {k: [float(x) for x in v] for k, v in history2.history.items()},
        "final_metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(CONFIG["history_save_path"], "w") as f:
        json.dump(all_history, f, indent=2)

    print(f"\nDone! Best model saved: {CONFIG['model_save_path']}")

if __name__ == "__main__":
    main()
