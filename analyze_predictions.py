import numpy as np
import cv2
import tensorflow as tf
import os
from pathlib import Path

# =========================
# 🤖 LOAD MODEL
# =========================
model_path = r"C:\Users\Hemalatha\OneDrive\Desktop\BrainTumor3D\models\segmentation_model.keras"
model = tf.keras.models.load_model(model_path)

# =========================
# 📁 DATA PATH
# =========================
data_dir = Path("data") / "images"

# Test images (you can change this list)
test_images = ['22.png', '1076.png', '1132.png', '1197.png', '1253.png', '1318.png']

print("=" * 80)
print("🧠 BRAIN TUMOR MODEL PREDICTION ANALYSIS")
print("=" * 80)

# =========================
# 🔁 LOOP IMAGES
# =========================
for img_name in test_images:

    img_path = data_dir / img_name

    if not img_path.exists():
        print(f"\n⚠ File not found: {img_name}")
        continue

    # =========================
    # 🖼 LOAD IMAGE
    # =========================
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"\n❌ Cannot read image: {img_name}")
        continue

    # =========================
    # 📏 PREPROCESS
    # =========================
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # =========================
    # 🤖 PREDICTION
    # =========================
    pred = model.predict(img_input, verbose=0)

    pred_mask = pred[0, :, :, 0]

    # =========================
    # 📊 ANALYSIS
    # =========================
    tumor_score = float(np.max(pred_mask))
    mean_score = float(np.mean(pred_mask))

    # =========================
    # 🎯 FINAL CLASSIFICATION
    # =========================
    if tumor_score >= 0.5:
        label = "🧠 Tumor Present"
    else:
        label = "✅ No Tumor"

    # =========================
    # 🖨 OUTPUT
    # =========================
    print(f"\n📌 {img_name}")
    print(f"Mean Score: {mean_score:.4f}")
    print(f"Max Score: {tumor_score:.4f}")
    print(f"FINAL RESULT: {label}")

print("\n" + "=" * 80)
print("✔ Analysis Completed")
print("=" * 80)