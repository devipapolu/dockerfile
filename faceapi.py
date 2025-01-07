from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
from mtcnn import MTCNN
from pymongo import MongoClient
import base64
import torch
from facenet_pytorch import InceptionResnetV1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU if you want CPU-only. Change to "0" or device index for GPU.
# print("Is CUDA available:", torch.cuda.is_available())
# print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# import tensorflow as tf
# print("Is GPU available:", tf.config.list_physical_devices('GPU'))

app = FastAPI()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client['face_data']
faces_collection = db['registered_faces']

# Load the InceptionResnetV1 model
# Load the InceptionResnetV1 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = InceptionResnetV1(pretrained=None, classify=False) 
model = InceptionResnetV1(pretrained=None, classify=False).to(device)
 # Load InceptionResnetV1

 

# Load the checkpoint and filter out unwanted keys
checkpoint = torch.load("20180402-114759-vggface2.pt", map_location=device)
filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("logits.")}
model.load_state_dict(filtered_checkpoint)  # Load the filtered state dictionary

model = model.to(device)
model.eval()

# Initialize MTCNN for face detection
detector = MTCNN()




# Helper function to convert base64 image to OpenCV image
def to_cv2_image(base64_str: str) -> np.ndarray:
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# Helper function to preprocess the face for the Torch model
def preprocess_face(face: np.ndarray) -> torch.Tensor:
    face = cv2.resize(face, (160, 160))  # Resize to 160x160 as required by InceptionResnetV1
    face = face / 255.0  # Normalize to [0, 1]
    face = (face - 0.5) / 0.5  # Normalize to [-1, 1]
    face = np.transpose(face, (2, 0, 1))  # Convert to CxHxW format
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    return face

class ImageData(BaseModel):
    base64_image: str

@app.post("/getFaceEmbeddings")
async def get_face_embeddings(image_data: ImageData):
    base64_image = image_data.base64_image

    if not base64_image:
        raise HTTPException(status_code=400, detail="No image data provided")

    image = to_cv2_image(base64_image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_image)

    if not detections:
        return {"face_locations": [], "face_embeddings": []}

    face_embeddings = []
    face_locations = []

    for detection in detections:
        x, y, width, height = detection['box']
        x1, y1, x2, y2 = max(0, x), max(0, y), x + width, y + height  # Ensure valid bounding box
        face = rgb_image[y1:y2, x1:x2]  # Crop face

        if face.size == 0:  # Handle invalid face crops
            continue

        preprocessed_face = preprocess_face(face)  # Preprocess the face

        with torch.no_grad():
            embedding = model(preprocessed_face).cpu().numpy().flatten()  # Generate embedding

        face_embeddings.append(embedding.tolist())  # Convert NumPy array to list
        face_locations.append([int(x1), int(y1), int(x2), int(y2)])  # Ensure bounding box values are integers

    return {"face_locations": face_locations, "face_embeddings": face_embeddings}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
