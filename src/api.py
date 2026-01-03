from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from config import BEST_MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

app = FastAPI(title="Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally to avoid reloading on every request
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        if tf.io.gfile.exists(BEST_MODEL_PATH):
            model = tf.keras.models.load_model(BEST_MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0) # Batch dimension
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    contents = await file.read()
    try:
        processed_image = preprocess_image(contents)
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # CIFAR-10 Classes (Update for custom)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        
        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Image Classification API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
