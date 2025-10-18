import os
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ Devices: {[d.device_type for d in tf.config.list_physical_devices()]}")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_DIR = "Library of Don_t Shots"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "image_quality_model.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

# ------------------------------------------------------------
# FEATURE EXTRACTOR (MobileNetV3)
# ------------------------------------------------------------
cnn = MobileNetV3Small(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = cnn.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"⚠️ Skipped {img_path}: {e}")
        return None

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
X, y, label_names = [], [], []
categories = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
print(f"\n🔍 Found categories: {categories}")

for idx, folder in enumerate(sorted(categories)):
    folder_path = os.path.join(DATA_DIR, folder)
    print(f"\n📂 Processing '{folder}' ...")
    for file in tqdm(os.listdir(folder_path), desc=f"{folder}", ncols=80):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            fpath = os.path.join(folder_path, file)
            feat = extract_features(fpath)
            if feat is not None:
                X.append(feat)
                y.append(idx)
    label_names.append(folder)

X, y = np.array(X), np.array(y)
print(f"\n✅ Loaded {len(X)} images across {len(label_names)} categories.")

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n🚀 Training RandomForestClassifier ...")
model = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------
print("\n📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test), target_names=label_names))

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(label_names, LABELS_PATH)
print(f"\n💾 Model saved → {MODEL_PATH}")
print(f"💾 Labels saved → {LABELS_PATH}")
print("\n✅ Training completed successfully.")
