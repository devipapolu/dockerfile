
import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from mongoengine import connect, Document, StringField, IntField
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import threading
import queue
import yaml
import os
import base64
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
import time

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Flask app setup
app = Flask(__name__)
CORS(app)
# Configuration
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        print("Configuration loaded successfully.")
except FileNotFoundError:
    print("Error: config.yaml file not found.")
    exit()

# MongoDB setup
client = MongoClient(config["db_connection_str"])
db = client[config["MONGO_DB_NAME"]]
attendance_logs = db[config["attendance_collection_name"]]
alerts = db[config["alert_collection_name"]]
employees = db[config["facedb_collection_name"]]
visitors = db[config["visitor_collection_name"]]
onboarding = db[config["onboarding_collection_name"]]


# print("Is CUDA available:", torch.cuda.is_available())
# print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Load InceptionResnetV1 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained=None, classify=False).to(device)

# model = InceptionResnetV1(pretrained=None, classify=False)
checkpoint = torch.load("20180402-114759-vggface2.pt", map_location=device)
filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("logits.")}
model.load_state_dict(filtered_checkpoint)
model = model.to(device).eval()

# Initialize MTCNN for face detection
detector = MTCNN(keep_all=True, device=device)


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




# Preload embeddings
cached_embeddings = {"EMPLOYEE": [], "VISITOR": [], "ONBOARDING": []}

for collection, collection_type in [(employees, "EMPLOYEE"), (visitors, "VISITOR"), (onboarding, "ONBOARDING")]:
    for record in collection.find():
        if 'embeddings' in record and record['embeddings']:
            embedding = np.array(record['embeddings'][0]['embedding'])
            if embedding.shape == (512,):  # Confirm embedding size
                cached_embeddings[collection_type].append({
                    "embedding": embedding,
                    "person_id": record["_id"],
                    "name": record["name"],
                    "image": record.get("base64_image", None)
                })
            else:
                print(f"Invalid embedding shape for {record['name']}: {embedding.shape}")



def preprocess_face(face):
    face = cv2.resize(face, (160, 160)) / 255.0
    face = (face - 0.5) / 0.5
    face = torch.tensor(np.transpose(face, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
    return face

def unlock_door(name):
    print(f"Door unlocked for {name}!")

def generate_tts(text):
    tts = gTTS(text, lang='en')
    tts_file = "greeting.mp3"
    tts.save(tts_file)
    os.system(f"mpg123 {tts_file}")

def send_self_registration_signal(person_type):
    message = {
        "action": "redirect",
        "form": "self_registration",
        "type": person_type.lower()
    }
    print(f"Signal sent to frontend for {person_type}: {message}")

def is_face_clear(face):
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian > 100

# Global variables
recognized_faces = []
unrecognized_faces = []
last_recognized_times = {}  
last_unrecognized_times = {}  

recognition_cooldown = 10
unrecognized_cooldown = 20
frame_skip = 10
frame_queue = queue.Queue(maxsize=10)

def recognize_and_log(frame):
    if frame is None or frame.size == 0:  # Check for empty or invalid frames
        print("Warning: Received an empty frame.")
        return

    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = detector.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = rgb_frame[y1:y2, x1:x2]
            if face.size == 0: continue

            preprocessed_face = preprocess_face(face)
            with torch.no_grad():
                embedding = model(preprocessed_face).numpy().flatten()

            recognized_name, person_id, person_type = "Unknown", None, "UNKNOWN"
            color = (0, 0, 255)
            is_recognized = False

            for collection_type, records in cached_embeddings.items():
                distances = [np.linalg.norm(embedding - rec["embedding"]) for rec in records]
                min_distance = min(distances) if distances else float("inf")
                
                if min_distance < 1.0:
                    record = records[distances.index(min_distance)]
                    recognized_name = record["name"]
                    person_id = record["person_id"]
                    person_type = collection_type
                    color = (0, 255, 0) if collection_type == "EMPLOYEE" else (0, 255, 255)
                    is_recognized = True
                    break

            if is_recognized:
                if person_id not in last_recognized_times or current_time - last_recognized_times[person_id] > recognition_cooldown:
                    last_recognized_times[person_id] = current_time
                    if person_type == "EMPLOYEE":
                        attendance_logs.update_one(
                            {"person_id": person_id},
                            {"$set": {"timestamp": datetime.now()}},
                            upsert=True
                        )
                    recognized_faces.append({
                        "person_id": str(person_id),
                        "name": recognized_name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "category": person_type.lower()
                    })
                    unlock_door(recognized_name)
                    generate_tts(f"Hi {recognized_name}, Welcome to Brihaspathi")
            else:
                face_key = f"{x1}_{y1}_{x2}_{y2}"
                if face_key not in last_unrecognized_times or current_time - last_unrecognized_times[face_key] > unrecognized_cooldown:
                    last_unrecognized_times[face_key] = current_time
                    _, jpeg_face = cv2.imencode('.jpg', face)
                    base64_face = base64.b64encode(jpeg_face).decode('utf-8')
                    unrecognized_faces.append({
                        "image": base64_face,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    alerts.insert_one({"type": "UNKNOWN", "timestamp": datetime.now(), "image": base64_face})
                    send_self_registration_signal("unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, recognized_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame        




@app.route('/')
def index():
    return render_template('index.html')


def frame_producer(cap):
    """Reads frames and puts them in a queue."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            frame_queue.get()  # Remove the oldest frame
            frame_queue.put(frame)


def frame_consumer():
    """Consumes frames, processes them, and streams them."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame = recognize_and_log(frame)  # Process the frame
            if processed_frame is not None:
                _, jpeg_frame = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(frame_consumer(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/audio_feed')
def audio_feed():
    def generate_audio():
        while True:
            with open("audio_stream.wav", "rb") as audio_file:
                audio_data = audio_file.read(1024)
                yield audio_data
    return Response(generate_audio(), mimetype="audio/wav")

@app.route('/recognized_faces')
def recognized_faces_endpoint():
    global recognized_faces
    return jsonify(recognized_faces)

@app.route('/unrecognized_faces')
def unrecognized_faces_endpoint():
    global unrecognized_faces
    return jsonify(unrecognized_faces)


def get_face_embeddings_from_api(base64_image):
    url = "http://localhost:8000/getFaceEmbeddings"  # Update to the host/port of your FastAPI
    payload = {"base64_image": base64_image}
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        raise Exception(f"Failed to get face embeddings: {response.text}")

    return response.json()

@app.route('/register_employee', methods=['POST'])
def register_employee():
    try:
        # Ensure the Content-Type is application/json
        if request.content_type != 'application/json':
            return jsonify({"message": "Content-Type must be application/json"}), 415

        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 400

        # Validate required fields
        required_fields = ['person_id', 'name', 'gender', 'image']
        for field in required_fields:
            if field not in data:
                return jsonify({"message": f"Missing required field: {field}"}), 400

        # Extract the base64 image from the request
        base64_image = data["image"]  ###convert and capture in frontend

        # Call the /getFaceEmbeddings API to get embeddings
        try:
            face_embedding_result = get_face_embeddings_from_api(base64_image)
        except Exception as e:
            return jsonify({"message": f"Failed to generate face embeddings: {str(e)}"}), 500

        # Extract face_embeddings
        face_embeddings = face_embedding_result.get("face_embeddings", [])

        if not face_embeddings:
            return jsonify({"message": "No faces detected in the provided image"}), 400

        # Save the image to a local folder
        image_url = f"registered_images/{data['person_id']}.jpg"
        image_path = os.path.join("registered_images", f"{data['person_id']}.jpg")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, "wb") as image_file:
            image_file.write(base64.b64decode(base64_image))

        # Create employee document
        employee_data = {
            "person_id": data["person_id"],
            "name": data["name"],
            "gender": data["gender"],
            "embeddings": face_embeddings[0],  # Assuming first face embedding
            "base64_image": base64_image,
            "image_url": image_url,
            "registered_on": datetime.utcnow().isoformat()
        }

        # Insert into MongoDB
        result = employees.insert_one(employee_data)

        # Convert the MongoDB ObjectId to a string
        employee_data["_id"] = str(result.inserted_id)

        # Return success response
        return jsonify({
            "message": "Employee registered successfully",
            "employee": employee_data
        }), 201

    except Exception as e:
        return jsonify({"message": f"Error registering employee: {str(e)}"}), 500







if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Read VIDEO_SOURCE from the config file; use 0 for webcam, 1 for RTSP
    video_source = config.get("VIDEO_SOURCE", 0)
    
    if video_source == 1:  # Use RTSP stream
        rtsp_url = config.get("rtsp_stream_url", "rtsp://default_url_here")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Error: Failed to open RTSP stream {rtsp_url}")
            exit()
        print(f"Using RTSP stream: {rtsp_url}")

    else:  # Use default webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Failed to open webcam.")
            exit()
        print("Using default webcam.")

    
    producer_thread = threading.Thread(target=frame_producer, args=(cap,))
    producer_thread.daemon = True
    producer_thread.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)
    cap.release()