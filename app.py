import os
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.applications import MobileNetV3Small  # type: ignore
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

# ----------------------------------------------------------
# FastAPI setup
# ----------------------------------------------------------
app = FastAPI(title="AI Image Quality Checker")
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
MODEL_PATH = "models/image_quality_model.pkl"
LABELS_PATH = "models/labels.pkl"

try:
    model = joblib.load(MODEL_PATH)
    label_names = joblib.load(LABELS_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    model, label_names = None, []

# ----------------------------------------------------------
# Feature Extractor
# ----------------------------------------------------------
cnn = MobileNetV3Small(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path: str):
    """Extract MobileNetV3 feature vector for the image."""
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

# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Render upload page."""
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "image_url": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)):
    """Handle image upload and classification."""
    if model is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": {"status": "❌ Error", "issue": "Model not loaded"}, "image_url": None}
        )

    # Save uploaded image
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract features
    feat = extract_features(file_path)
    if feat is None:
        result = {"status": "❌ Error", "issue": "Invalid or unreadable image", "confidence": "N/A"}
    else:
        probs = model.predict_proba([feat])[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        reason = label_names[pred_idx]

        # Define confidence threshold for failure detection
        THRESHOLD = 0.55

        if confidence >= THRESHOLD:
            result = {"status": "❌ FAIL", "issue": reason, "confidence": f"{confidence * 100:.2f}%"}
        else:
            result = {"status": "✅ PASS", "issue": "No issues detected", "confidence": f"{confidence * 100:.2f}%"}

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "image_url": f"/uploads/{file.filename}"}
    )
