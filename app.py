
import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from mongoengine import connect, Document, StringField, IntField
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
import requests
import queue
import random
import yaml
import os
import base64
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
import time
from threading import Thread
import threading
import uvicorn
import jwt
from datetime import datetime, timedelta 
# import datetime
from bson import ObjectId
from bson.json_util import dumps


# # Secret key for JWT
SECRET_KEY = "sdfghjklcvgbhnm"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


fastapi_app = FastAPI()

# Flask app setup
flask_app = Flask(__name__)
CORS(flask_app)



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
users = db[config["users_collection_name"]]


print("Is CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

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

@fastapi_app.post("/getFaceEmbeddings")
async def get_face_embeddings(image_data: ImageData):
    base64_image = image_data.base64_image

    if not base64_image:
        raise HTTPException(status_code=400, detail="No image data provided")

    # Convert base64 string to CV2 image
    image = to_cv2_image(base64_image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MTCNN's detect method to detect faces
    boxes, _ = detector.detect(rgb_image)

    if boxes is None or len(boxes) == 0:
        return {"face_locations": [], "face_embeddings": []}

    face_embeddings = []
    face_locations = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # Ensure valid bounding box
        face = rgb_image[y1:y2, x1:x2]  # Crop the face

        if face.size == 0:  # Handle invalid face crops
            continue

        preprocessed_face = preprocess_face(face)  # Preprocess the face

        with torch.no_grad():
            embedding = model(preprocessed_face).cpu().numpy().flatten()  # Generate embedding

        face_embeddings.append(embedding.tolist())  # Convert NumPy array to list
        face_locations.append([x1, y1, x2, y2])  # Save bounding box as integers

    return {"face_locations": face_locations, "face_embeddings": face_embeddings}




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cached_embeddings = {"EMPLOYEE": [], "VISITOR": []}

def print_cached_embeddings_info():
    if 'EMPLOYEE' in cached_embeddings and 'VISITOR' in cached_embeddings:
        logging.info(f"Number of cached embeddings for EMPLOYEES: {len(cached_embeddings['EMPLOYEE'])}")
        logging.info(f"Number of cached embeddings for VISITORS: {len(cached_embeddings['VISITOR'])}")
    else:
        logging.error("Cached embeddings categories are not properly initialized.")

def fetch_and_update_cache():
    while True:
        new_cache = {"EMPLOYEE": [], "VISITOR": []}
        try:
            for collection, collection_type in [(employees, "EMPLOYEE"), (visitors, "VISITOR")]:
                for record in collection.find():
                    if 'embeddings' in record and record['embeddings']:
                        embedding = np.array(record['embeddings'][0]['embedding'])
                        if embedding.shape == (512,):
                            new_cache[collection_type].append({
                                "embedding": embedding,
                                "mongo_id": str(record["_id"]),
                                "name": record.get("name", "Unknown"),
                                "unique_id": str(record.get("unique_id", "None")),
                                "image": record.get("base64_image", None)
                            })
                        else:
                            logging.warning(f"Invalid embedding shape for {record.get('name', 'Unknown')}: {embedding.shape}")
            global cached_embeddings
            cached_embeddings = new_cache
            print_cached_embeddings_info()
        except Exception as e:
            logging.error(f"Error updating cached embeddings: {str(e)}")
        time.sleep(20)  # Poll every 20 seconds

def start_polling():
    polling_thread = threading.Thread(target=fetch_and_update_cache)
    polling_thread.daemon = True
    polling_thread.start()

# Debug: Print keys of the first record for each collection type
# for collection_type, records in cached_embeddings.items():
#     if records:  # Check if there are records
#         print(f"Keys of the first cached embedding for {collection_type}: {list(records[0].keys())}")
#         break  # Exit after printing the first collection's keys





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
    os.system(f"mpg123 -q {tts_file}")  # The -q flag suppresses output


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
frame_count = 0
frame_queue = queue.Queue(maxsize=50)


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
            if face.size == 0:
                continue

            preprocessed_face = preprocess_face(face)
            with torch.no_grad():
                embedding = model(preprocessed_face).cpu().numpy().flatten()

            recognized_name, unique_id, person_type = "Unknown", None, "UNKNOWN"
            color = (0, 0, 255)
            is_recognized = False

            for collection_type, records in cached_embeddings.items():
                distances = [np.linalg.norm(embedding - rec["embedding"]) for rec in records]
                min_distance = min(distances) if distances else float("inf")
                

                if min_distance < 1.0:  # Threshold for recognition
                    record = records[distances.index(min_distance)]
                    recognized_name = record.get("name", "Unknown")
                    unique_id = record.get("unique_id", None)
                    person_type = collection_type
                    color = (0, 255, 0) if collection_type == "EMPLOYEE" else (0, 255, 255)
                    is_recognized = True
                    break

            if is_recognized:
                if unique_id not in last_recognized_times or current_time - last_recognized_times[unique_id] > recognition_cooldown:
                    last_recognized_times[unique_id] = current_time
                    if person_type == "EMPLOYEE":
                        attendance_logs.update_one(
                            {"unique_id": unique_id},
                            {"$set": {"timestamp": datetime.now()}},
                            upsert=True
                        )
                    
                    base64_image = record.get("image", None)

                    recognized_face = {
                        "unique_id": str(unique_id),
                        "name": recognized_name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "category": person_type.lower(),
                        "image": base64_image,  # Add base64 image
                        # "recognition_time": current_time 
                    }
                    recognized_faces.append(recognized_face)


                    print("Keys of appended data in recognized_faces:", list(recognized_face.keys()))                    

                    unlock_door(recognized_name)
                    generate_tts(f"Hi {recognized_name}, Welcome to Brihaspathi")
            else:
                face_key = f"{x1}_{y1}_{x2}_{y2}"
                if face_key not in last_unrecognized_times or current_time - last_unrecognized_times[face_key] > unrecognized_cooldown:
                    last_unrecognized_times[face_key] = current_time
                    _, jpeg_face = cv2.imencode('.jpg', face)
                    base64_face = base64.b64encode(jpeg_face).decode('utf-8')
                    unrecognized_face = {
                        "image": base64_face,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    unrecognized_faces.append(unrecognized_face)
                    # print("Added to unrecognized_faces:", unrecognized_face)  # Debug
                    alerts.insert_one({"type": "UNKNOWN", "timestamp": datetime.now(), "image": base64_face})
                    send_self_registration_signal("unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, recognized_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame





@flask_app.route('/')
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
    global frame_count  # Use the global frame_count variable
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_count += 1
            if frame_count % frame_skip == 0:  # Process every 'frame_skip' frames
                processed_frame = recognize_and_log(frame)  # Process the frame
                if processed_frame is not None:
                    # Compress frame to JPEG with reduced quality for faster streaming
                    _, jpeg_frame = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')

@flask_app.route('/video_feed')
def video_feed():
    return Response(frame_consumer(), mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/audio_feed')
def audio_feed():
    def generate_audio():
        while True:
            with open("audio_stream.wav", "rb") as audio_file:
                audio_data = audio_file.read(1024)
                yield audio_data
    return Response(generate_audio(), mimetype="audio/wav")

@flask_app.route('/recognized_faces')
def recognized_faces_endpoint():
    global recognized_faces
    return jsonify(recognized_faces)



@flask_app.route('/unrecognized_faces')
def unrecognized_faces_endpoint():
    global unrecognized_faces
    return jsonify(unrecognized_faces)



def get_face_embeddings_from_api(base64_images):
    url = "http://localhost:8000/getFaceEmbeddings"  # Update to the host/port of your FastAPI
    embeddings = []
    
    for base64_image in base64_images:
        payload = {"base64_image": base64_image}
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            raise Exception(f"Failed to get face embeddings: {response.text}")

        face_data = response.json()
        if "face_embeddings" in face_data and face_data["face_embeddings"]:
            embeddings.append(face_data["face_embeddings"][0])  # Use the first embedding

    return embeddings




    
@flask_app.route('/register_employee', methods=['POST'])
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
        required_fields = ['unique_id', 'name', 'gender', 'designation', 'email_id', 'phone_number', 'images']
        for field in required_fields:
            if field not in data:
                return jsonify({"message": f"Missing required field: {field}"}), 401

        # Check for existing records with the same unique_id or email_id
        existing_unique_id = employees.find_one({"unique_id": data['unique_id']})
        if existing_unique_id:
            return jsonify({"message": "Unique ID already exists", "unique_id":True}), 209
        
        existing_email = employees.find_one({"email_id": data['email_id']})
        if existing_email:
            return jsonify({"message": "Email ID already exists", "email_id":True}), 209

        existing_name = employees.find_one({"name": data['name']})
        if existing_name:
            return jsonify({"message": "Name already exists", "name":True}), 209
        

        # Extract fields from the request
        base64_images = data["images"]
        if not base64_images or not isinstance(base64_images, list):
            return jsonify({"message": "Invalid images format. Provide a list of base64-encoded images."}), 402

        # Call the /getFaceEmbeddings API to get embeddings for all images
        try:
            face_embeddings = get_face_embeddings_from_api(base64_images)
        except Exception as e:
            return jsonify({"message": f"Failed to generate face embeddings: {str(e)}", "embeddingerror": True}), 204

        if not face_embeddings:
            return jsonify({"message": "No faces detected in the provided images", "nofaceerror": True}), 203

        # Save the first image to a local folder
        image_url = f"registered_images/{data['unique_id']}.jpg"
        image_path = os.path.join("registered_images", f"{data['unique_id']}.jpg")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, "wb") as image_file:
            image_file.write(base64.b64decode(base64_images[0]))  # Save the first image

        # Create employee document
        employee_data = {
            "unique_id": data["unique_id"],
            "name": data["name"],
            "gender": data["gender"],
            "designation": data["designation"],
            "email_id": data["email_id"],
            "phone_number": data["phone_number"],
            "embeddings": [{"angle": "unspecified", "embedding": embedding} for embedding in face_embeddings],
            "base64_image": base64_images[0],  # Store the first base64 image
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
            "success": True
        }), 201

    except Exception as e:
        return jsonify({"message": f"Error registering employee: {str(e)}"}), 500



def generate_unique_id():
    """Generate a unique 4-digit ID."""
    while True:
        unique_id = str(random.randint(1000, 9999))  # Generate a random 4-digit number
        if not visitors.find_one({"unique_id": unique_id}):  # Ensure the ID is unique
            return unique_id

@flask_app.route('/self_registration', methods=['POST'])
def self_registration():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 402

        # Validate required fields
        required_fields = [
            'category', 'name', 'gender', 'purpose', 
            'meet_employee', 'images', 'phone_number', 'email_id', 
            'signature', 'created_by','address'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"message": f"Missing required field: {field}"}), 400

        # Extract data fields
        category = data["category"]
        name = data["name"]
        gender = data["gender"]
        purpose = data["purpose"]
        meet_employee = data["meet_employee"]
        phone_number = data["phone_number"]
        email_id = data["email_id"]
        signature = data["signature"]
        base64_images = data["images"]
        address = data["address"]
        created_by = data["created_by"]

        if not base64_images or not isinstance(base64_images, list):
            return jsonify({"message": "Invalid images format. Provide a list of base64-encoded images."}), 402

        # Check if the selected employee exists in the database
        employee = employees.find_one({"name": meet_employee})
        if not employee:
            return jsonify({"message": f"Employee '{meet_employee}' not found."}), 403

        # Call the /getFaceEmbeddings API to get embeddings for all images
        try:
            face_embeddings = get_face_embeddings_from_api(base64_images)
        except Exception as e:
            return jsonify({"message": f"Failed to generate face embeddings: {str(e)}"}), 500

        if not face_embeddings:
            return jsonify({"message": "No faces detected in the provided images"}), 404

        # Generate a unique ID for the visitor
        unique_id = generate_unique_id()

        # Create self-registration document
        registration_data = {
            "category": category,
            "name": name,
            "gender": gender,
            "purpose": purpose,
            "phone_number": phone_number,
            "email_id": email_id,
            "meet_employee": meet_employee,
            "embeddings": [{"angle": "unspecified", "embedding": embedding} for embedding in face_embeddings],
            "base64_image": base64_images[0],
            "signature_base64": signature,
            "checkin_time": "",
            "checkout_time": "",
            "status": "pending",
            "created_by": created_by,
            "address": address,
            "unique_id": unique_id,  # Add unique ID
            "registered_on": datetime.utcnow()
        }

        # Insert into the visitor collection
        result = visitors.insert_one(registration_data)

        # Convert the MongoDB ObjectId to a string
        registration_data["_id"] = str(result.inserted_id)

        # Return success response
        return jsonify({
            "message": "Self-registration completed successfully",
            "unique_id": unique_id,  # Include the unique ID in the response
            "success": True
        }), 201

    except Exception as e:
        return jsonify({"message": f"Error during self-registration: {str(e)}"}), 500


@flask_app.route('/register_user', methods=['POST'])
def register_user():
    try:
        # Ensure the Content-Type is application/json
        if request.content_type != 'application/json':
            return jsonify({"message": "Content-Type must be application/json"}), 415

        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 403

        # Validate required fields
        required_fields = ['name', 'email_id', 'password', 'base64_image', 'role', 
                           'phone_number', 'designation', 'unique_id', 'primary_id']
        for field in required_fields:
            if field not in data:
                return jsonify({"message": f"Missing required field: {field}"}), 400

        # Extract and validate fields
        name = data['name']
        email_id = data['email_id']
        password = data['password']
        base64_image = data['base64_image']
        role = data['role']
        phone_number = data['phone_number']
        designation = data['designation']
        unique_id = data['unique_id']
        primary_id = data['primary_id']

        # Ensure the email is unique
        existing_user = db.users.find_one({"email_id": email_id})
        if existing_user:
            return jsonify({"message": "Email already exists"}), 400

       
        # Save the image to a local folder
        image_url = f"user_images/{unique_id}.jpg"
        image_path = os.path.join("user_images", f"{unique_id}.jpg")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        with open(image_path, "wb") as image_file:
            image_file.write(base64.b64decode(base64_image))  # Save the image

        # Create the user document
        user_data = {
            "name": name,
            "email_id": email_id,
            "password": password,
            "base64_image": base64_image,
            "role": role,
            "phone_number": phone_number,
            "designation": designation,
            "unique_id": unique_id,
            "primary_id": primary_id,
            "image_url": image_url,
            "registered_on": datetime.utcnow().isoformat()
        }

        # Insert the user into MongoDB
        result = db.users.insert_one(user_data)

        # Convert the MongoDB ObjectId to a string
        user_data["_id"] = str(result.inserted_id)

        # Return success response
        return jsonify({
            "message": "User registered successfully",
            # "user": user_data
            "success":True
        }), 201

    except Exception as e:
        return jsonify({"message": f"Error registering user: {str(e)}"}), 500




@flask_app.route('/sign_in', methods=['POST'])
def sign_in():
    try:
        # Ensure the Content-Type is application/json
        # if request.content_type != 'application/json':
        #     return jsonify({"message": "Content-Type must be application/json"}), 415

        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 400

        # Validate required fields
        required_fields = ['email_id', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({"message": f"Missing required field: {field}"}), 400
            

        # Extract email and password
        email_id = data['email_id']
        password = data['password']

        # Find the user by email
        user = users.find_one({"email_id": email_id})
        if not user:
            return jsonify({"message": "Invalid email or password"}), 205

       # Compare passwords directly
        if user['password'] != password:
            return jsonify({"message": "Invalid email or password"}), 205

        # Generate JWT token without expiration
        token_payload = {
            "user_id": str(user["_id"]),
            "email_id": user["email_id"],
            "role": user["role"],
            "base64_image":user["base64_image"],
            "name":user["name"]
        }
        token = jwt.encode(token_payload, SECRET_KEY, algorithm="HS256")

        # Return success response with token
        return jsonify({
            "message": "Sign-in successful",
            "success":True,
            "token": token,
            "role":user["role"],
            "user": {
                "id": str(user["_id"]),
                "name": user["name"],
                "email_id": user["email_id"],
                "role": user["role"],
            }
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error during sign-in: {str(e)}"}), 500

@flask_app.route('/decode_token', methods=['POST'])
def decode_token():
    try:
        # # Ensure the Content-Type is application/json
        # if request.content_type != 'application/json':
        #     return jsonify({"message": "Content-Type must be application/json"}), 415

        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 400

        # Validate required fields
        if 'token' not in data:
            return jsonify({"message": "Missing required field: token"}), 400

        token = data['token']

        # Decode the JWT token
        try:
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401

        # Return the decoded token
        return jsonify({
            "message": "Token decoded successfully",
            "decoded_token": decoded_token
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error decoding token: {str(e)}"}), 500


@flask_app.route('/get_all_visitors', methods=['GET'])
def get_all_visitors():
    try:
        # Retrieve all visitor documents from the collection
        visitors_data = visitors.find()
        
        # Convert MongoDB cursor to a list of dictionaries
        visitor_list = []
        for visitor in visitors_data:
            visitor["_id"] = str(visitor["_id"])  # Convert ObjectId to string
            visitor_list.append(visitor)
        
        return jsonify({
            "message": "Visitors retrieved successfully",
            "visitors": visitor_list
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error retrieving visitors: {str(e)}"}), 500

@flask_app.route('/edit_visitor/<visitor_id>', methods=['PUT'])

def edit_visitor(visitor_id):
    try:
        # Ensure the Content-Type is application/json
        if request.content_type != 'application/json':
            return jsonify({"message": "Content-Type must be application/json"}), 415

        # Parse the visitor ID
        from bson import ObjectId
        try:
            visitor_id = ObjectId(visitor_id)
        except Exception:
            return jsonify({"message": "Invalid visitor ID format."}), 400

        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 400

        # Check if the visitor exists
        visitor = visitors.find_one({"_id": visitor_id})
        if not visitor:
            return jsonify({"message": f"Visitor with ID '{visitor_id}' not found."}), 404

        # Fields that can be updated (excluding image, signature, checkin_time, checkout_time, and status)
        editable_fields = [
            'name', 'gender', 'purpose', 'meet_employee', 
            'phone_number', 'email_id', 'created_by', 'category', 'address'
        ]

        # Prepare the update data
        update_data = {}
        for field in editable_fields:
            if field in data:
                update_data[field] = data[field]

        # Validate if fields are present in the request
        if not update_data:
            return jsonify({"message": "No editable fields provided for update."}), 400

        # Validate the category if provided
        # if 'category' in update_data and update_data['category'] not in ['visitor']:
        #     return jsonify({"message": "Invalid category. Must be 'visitor'."}), 400

        # Check if the selected employee exists in the database for meet_employee field
        if 'meet_employee' in update_data:
            employee = employees.find_one({"name": update_data["meet_employee"]})
            if not employee:
                return jsonify({"message": f"Employee '{update_data['meet_employee']}' not found."}), 400

        # Perform the update
        result = visitors.update_one({"_id": visitor_id}, {"$set": update_data})
        if result.modified_count == 0:
            return jsonify({"message": "No changes made to the visitor record."}), 200

        # Return success response
        return jsonify({
            "message": "Visitor updated successfully",
            "updated_fields": update_data
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error during visitor update: {str(e)}"}), 500
    

@flask_app.route('/get_all_employees', methods=['GET'])
def get_all_employees():
    try:
        # Retrieve all visitor documents from the collection
        employees_data = employees.find()
        
        # Convert MongoDB cursor to a list of dictionaries
        employees_list = []
        for employee in employees_data:
            employee["_id"] = str(employee["_id"])  # Convert ObjectId to string
            employees_list.append(employee)
        
        return jsonify({
            "message": "Employees retrieved successfully",
            "visitors": employees_list
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error retrieving visitors: {str(e)}"}), 500




@flask_app.route('/update_checkin/<visitor_id>', methods=['PUT'])
def update_checkin(visitor_id):
    try:
        # Ensure the Content-Type is application/json
        # if request.content_type != 'application/json':
        #     return jsonify({"message": "Content-Type must be application/json"}), 415

        # Validate the visitor ID format
        try:
            visitor_id = ObjectId(visitor_id)
        except Exception:
            return jsonify({"message": "Invalid visitor ID format."}), 400

        # Find the visitor in the database
        visitor = visitors.find_one({"_id": visitor_id})
        if not visitor:
            return jsonify({"message": f"Visitor with ID '{visitor_id}' not found."}), 404

        # Prepare update data
        update_data = {
            "checkin_time": datetime.utcnow(),
            "status": "check-in"
        }

        # Perform the update
        result = visitors.update_one({"_id": visitor_id}, {"$set": update_data})
        if result.modified_count == 0:
            return jsonify({"message": "No changes made to the visitor record."}), 200

        # Debug: Fetch updated document
        updated_visitor = visitors.find_one({"_id": visitor_id})
        updated_visitor["_id"] = str(updated_visitor["_id"]) 

        # Return success response
        return jsonify({
            "message": "Visitor check-in updated successfully.",
            "updated_fields": update_data,
            "updated_visitor": updated_visitor
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error during check-in update: {str(e)}"}), 500



@flask_app.route('/update_checkout/<visitor_id>', methods=['PUT'])
def update_checkout(visitor_id):
    try:
        # Ensure the Content-Type is application/json
        # if request.content_type != 'application/json':
        #     return jsonify({"message": "Content-Type must be application/json"}), 415

        # Validate the visitor ID format
        try:
            visitor_id = ObjectId(visitor_id)
        except Exception:
            return jsonify({"message": "Invalid visitor ID format."}), 400

        # Find the visitor in the database
        visitor = visitors.find_one({"_id": visitor_id})
        if not visitor:
            return jsonify({"message": f"Visitor with ID '{visitor_id}' not found."}), 404

        # Prepare update data
        update_data = {
            "checkout_time": datetime.utcnow(),
            "status": "check-out"
        }

        # Perform the update
        result = visitors.update_one({"_id": visitor_id}, {"$set": update_data})
        if result.modified_count == 0:
            return jsonify({"message": "No changes made to the visitor record."}), 200

        # Debug: Fetch updated document
        updated_visitor = visitors.find_one({"_id": visitor_id})
        updated_visitor["_id"] = str(updated_visitor["_id"]) 

        # Return success response
        return jsonify({
            "message": "Visitor check-out updated successfully.",
            "updated_fields": update_data,
            "updated_visitor": updated_visitor
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error during check-out update: {str(e)}"}), 500


@flask_app.route('/get_visitor/<visitor_id>', methods=['GET'])
def get_visitor_by_id(visitor_id):
    try:
        # Validate the visitor ID format
        try:
            visitor_id = ObjectId(visitor_id)
        except Exception:
            return jsonify({"message": "Invalid visitor ID format."}), 400

        # Fetch the visitor document
        visitor = visitors.find_one({"_id": visitor_id})
        if not visitor:
            return jsonify({"message": f"Visitor with ID '{visitor_id}' not found."}), 404

        # Convert the MongoDB ObjectId to a string for the response
        visitor["_id"] = str(visitor["_id"])

        # Return the visitor document
        return jsonify({
            "message": "Visitor retrieved successfully.",
            "visitor": visitor
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error fetching visitor: {str(e)}"}), 500


@flask_app.route('/get_employee/<employee_id>', methods=['GET'])
def get_employee_by_id(employee_id):
    try:
        # Validate the visitor ID format
        try:
            employee_id = ObjectId(employee_id)
        except Exception:
            return jsonify({"message": "Invalid visitor ID format."}), 400

        # Fetch the visitor document
        employee = employees.find_one({"_id": employee_id})
        if not employee:
            return jsonify({"message": f"Employee with ID '{employee_id}' not found."}), 404

        # Convert the MongoDB ObjectId to a string for the response
        employee["_id"] = str(employee["_id"])

        # If you need to manipulate the data into a list of objects
        # Here assuming you want to wrap the dictionary in a list
        response_data = [employee]  # This makes it an array of one object

        # Return the employee document in an array format
        return jsonify({
            "message": "Employee retrieved successfully.",
            "employee": response_data
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error fetching employee: {str(e)}"}), 500
    

@flask_app.route('/update_visitor/<string:visitor_id>', methods=['PUT'])
def update_visitor(visitor_id):
    try:
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data."}), 400

        # Validate visitor ID
        if not ObjectId.is_valid(visitor_id):
            return jsonify({"message": "Invalid visitor ID."}), 400

        # Fields allowed to update
        updatable_fields = ["name", "phone_number", "address", "purpose", "meet_employee"]

        # Filter valid fields from request data
        update_data = {key: value for key, value in data.items() if key in updatable_fields}

        # Check if there's any data to update
        if not update_data:
            return jsonify({"message": "No valid fields to update."}), 400

        # Attempt to find and update the visitor
        result = visitors.update_one(
            {"_id": ObjectId(visitor_id)},
            {"$set": update_data}
        )

        if result.matched_count == 0:
            return jsonify({"message": f"Visitor with ID {visitor_id} not found."}), 404

        # Return success message
        return jsonify({"message": "Visitor updated successfully."}), 200

    except Exception as e:
        return jsonify({"message": f"Error updating visitor: {str(e)}"}), 500


@flask_app.route('/get_user/<user_id>', methods=['GET'])
def get_user_by_id(user_id):
    try:
        # Validate the visitor ID format
        try:
            user_id = ObjectId(user_id)
        except Exception:
            return jsonify({"message": "Invalid visitor ID format."}), 400

        # Fetch the visitor document
        user = users.find_one({"_id": user_id})
        if not user:
            return jsonify({"message": f"user with ID '{user_id}' not found."}), 404

        # Convert the MongoDB ObjectId to a string for the response
        user["_id"] = str(user["_id"])

        # Return the user document
        return jsonify({
            "message": "user retrieved successfully.",
            "user": user
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error fetching visitor: {str(e)}"}), 500


@flask_app.route('/get_all_users', methods=['GET'])
def get_all_users():
    try:
        # Fetch all users from the users collection
        users_list = list(users.find())

        # Convert ObjectId to string for JSON serialization
        for user in users_list:
            user["_id"] = str(user["_id"])

        # Return the list of users
        return jsonify({
            "message": "Users retrieved successfully.",
            "users": users_list
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error fetching users: {str(e)}"}), 500


@flask_app.route('/edit_employee/<string:employee_id>', methods=['PUT'])
def edit_employee(employee_id):
    try:
        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 400

        # Editable fields based on frontend
        editable_fields = [
            'name', 'email_id', 'phone_number', 'unique_id', 'designation', 'base64_image'
        ]

        # Prepare update data
        update_data = {field: data[field] for field in editable_fields if field in data and data[field]}

        if not update_data:
            return jsonify({"message": "No valid fields to update."}), 400

        # Check for duplicate email_id and unique_id
        if 'email_id' in update_data:
            existing_email = employees.find_one(
                {"email_id": update_data["email_id"], "_id": {"$ne": ObjectId(employee_id)}}
            )
            if existing_email:
                return jsonify({"message": "Email ID already exists.", "success": False}), 209

        if 'unique_id' in update_data:
            existing_unique_id = employees.find_one(
                {"unique_id": update_data["unique_id"], "_id": {"$ne": ObjectId(employee_id)}}
            )
            if existing_unique_id:
                return jsonify({"message": "Unique ID already exists.", "success": False}), 209

        # Update the employee record in MongoDB
        result = employees.update_one({"_id": ObjectId(employee_id)}, {"$set": update_data})

        if result.matched_count == 0:
            return jsonify({"message": f"Employee with ID {employee_id} not found."}), 404

        # Return success response
        return jsonify({"message": "Employee updated successfully.", "success": True}), 200

    except Exception as e:
        return jsonify({"message": f"Error updating employee: {str(e)}"}), 500


@flask_app.route('/edit_user/<string:primary_id>', methods=['PUT'])
def edit_user(primary_id):
    try:
        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"message": "Invalid JSON data"}), 400

        # Editable fields based on frontend
        editable_fields = [
            'name', 'email_id', 'phone_number', 'unique_id', 'designation', 'base64_image'
        ]

        # Prepare update data
        update_data = {field: data[field] for field in editable_fields if field in data and data[field]}

        if not update_data:
            return jsonify({"message": "No valid fields to update."}), 400

        # Update the user record in MongoDB using primary_id
        result = users.update_one({"primary_id": primary_id}, {"$set": update_data})

        if result.matched_count == 0:
            return jsonify({"message": f"User with primary ID {primary_id} not found.",}), 204

        # Return success response
        return jsonify({"message": "User updated successfully.", "success": True}), 200

    except Exception as e:
        return jsonify({"message": f"Error updating user: {str(e)}"}), 500



@flask_app.route('/delete_employee/<string:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    try:
        result = employees.delete_one({"_id": ObjectId(employee_id)})
        if result.deleted_count == 0:
            return jsonify({"message": f"Employee with ID {employee_id} not found."}), 404
        return jsonify({"message": "Employee deleted successfully.","success":True}), 200
    except Exception as e:
        return jsonify({"message": f"Error deleting employee: {str(e)}"}), 500

   
@flask_app.route('/delete_user/<string:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        result = users.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            return jsonify({"message": f"User with ID {user_id} not found."}), 404
        return jsonify({"message": "User deleted successfully.", "success": True}), 200
    except Exception as e:
        return jsonify({"message": f"Error deleting user: {str(e)}"}), 500

@flask_app.route('/delete_user_byprimary/<string:primary_id>', methods=['DELETE'])
def delete_userbyprimary(primary_id):
    try:
        result = users.delete_one({"primary_id": primary_id})
        if result.deleted_count == 0:
            return jsonify({"message": f"User with ID {primary_id} not found."}), 204
        return jsonify({"message": "User deleted successfully.", "success": True}), 200
    except Exception as e:
        return jsonify({"message": f"Error deleting user: {str(e)}"}), 500




@flask_app.route('/delete_visitor/<string:visitor_id>', methods=['DELETE'])
def delete_visitor(visitor_id):
    try:
        result = visitors.delete_one({"_id": ObjectId(visitor_id)})
        if result.deleted_count == 0:
            return jsonify({"message": f"Visitor with ID {visitor_id} not found."}), 404
        return jsonify({"message": "Visitor deleted successfully.", "success":True}), 200
    except Exception as e:
        return jsonify({"message": f"Error deleting visitor: {str(e)}"}), 500



@flask_app.route('/visitors/by_employee/<employee_name>', methods=['GET'])
def get_visitors_by_employee_name(employee_name):
    """Fetch visitors that an employee has met."""
    try:
        # Find all visitors where 'meet_employee' matches the 'employee_name'
        visitors_cursor = visitors.find({"meet_employee": employee_name})
        visitors_list = list(visitors_cursor)

        # Process visitors to serialize MongoDB ObjectId
        for visitor in visitors_list:
            visitor['_id'] = str(visitor['_id'])

        # Always return a JSON object with a visitors array
        return jsonify({
            "message": "Visitors retrieved successfully for " + employee_name,
            "visitors": visitors_list
        }), 200

    except Exception as e:
        return jsonify({"error": "Error fetching data: " + str(e)}), 500


@flask_app.route('/get_userbyprimary/<string:primary_id>', methods=['GET'])
def get_user_by_primary_id(primary_id):
    try:
        # Find the user with the given primary_id
        user = users.find_one({"primary_id": primary_id})
        
        if not user:
            # Return an empty response if no user is found
            return jsonify({"message": "No user found", 
                            "success":False,
                            "user": {}}), 200
        
        # Convert ObjectId to string and prepare the response
        user["_id"] = str(user["_id"])
        return jsonify({"message": "User retrieved successfully.",
                        "success":True,
                         "user": user}), 200
    except Exception as e:
        return jsonify({"message": f"Error retrieving user: {str(e)}"}), 500
    




@flask_app.route('/recognized_visitor_faces', methods=['GET'])
def get_recognized_visitor_faces():
    try:
        # Filter recognized_faces for visitor category only
        recognized_visitors = [
            face for face in recognized_faces
            if face.get("category") == "visitor"
        ]

        visitors_with_details = []
        for face in recognized_visitors:
            unique_id = face.get("unique_id")
            # recognition_time = face.get("timestamp")  # Extract recognition_time
            if unique_id:
                # Fetch visitor data from the database
                visitor_data = visitors.find_one({"unique_id": unique_id})
                if visitor_data:
                    visitors_with_details.append({
                        "name": visitor_data.get("name", "Unknown"),
                        "phone_number": visitor_data.get("phone_number", "Unknown"),
                        "unique_id": visitor_data.get("unique_id", "Unknown"),
                        "image": visitor_data.get("base64_image", None),
                        "timestamp": face.get("timestamp", "Unknown") # Include recognition_time
                    })

        return jsonify({
            "message": "Recognized visitor faces retrieved successfully.",
            "recognized_visitors": visitors_with_details
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error retrieving recognized visitor faces: {str(e)}"}), 500

@flask_app.route('/resetpassword/<user_id>', methods=['PUT'])
def reset_password(user_id):
    try:
        # Parse the request JSON
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Invalid JSON data"}), 400

        old_password = data.get("oldPassword")
        new_password = data.get("password")

        # Check if required fields are present
        if not old_password or not new_password:
            return jsonify({"success": False, "message": "Old password and new password are required"}), 400

        # Fetch the user from the database
        user = users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404

        # Verify the old password directly
        if user["password"] != old_password:
            return jsonify({"success": False, "message": "Old password is incorrect"}), 205

        # Check if the new password is the same as the old password
        if user["password"] == new_password:
            return jsonify({"oldsame": True, "message": "New password cannot be the same as the old password"}), 204

        # Update the user's password in the database
        users.update_one({"_id": ObjectId(user_id)}, {"$set": {"password": new_password}})

        return jsonify({"success": True, "message": "Password updated successfully"}), 200

    except Exception as e:
        return jsonify({"success": False, "message": f"Error updating password: {str(e)}"}), 500

@flask_app.route('/visitor/<unique_id>', methods=['GET'])
def get_visitor(unique_id):
    visitor = visitors.find_one({"unique_id": unique_id})
    if visitor:
        return dumps(visitor), 200
    else:
        return jsonify({'error': 'Visitor not found'}), 404

def run_flask():
    # Start Flask app on port 5000
    flask_app.run(host="0.0.0.0", port=5000, threaded=True)

def run_fastapi():
    # Start FastAPI app on port 8000
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)


def run_video_capture():
    global cap
    # Read VIDEO_SOURCE from the config file
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

    # Set a lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480
    print("Webcam resolution set to 640x480.")

    # Start the producer thread for frame capture
    producer_thread = threading.Thread(target=frame_producer, args=(cap,))
    producer_thread.daemon = True
    producer_thread.start()


 # Make sure to import timedelta directly

def cleanup_old_alerts():
    retention_period = timedelta(minutes=30)  # Retain alerts for 30 minutes
    while True:
        cutoff_time = datetime.now() - retention_period  # Get the current local time minus the retention period
        deleted_count = alerts.delete_many({"timestamp": {"$lt": cutoff_time}}).deleted_count
        print(f"Cleaned up {deleted_count} old alerts older than {cutoff_time}")
        time.sleep(30)  # Check every 30 seconds for testing










if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize and start all required threads
        flask_thread = Thread(target=run_flask)
        fastapi_thread = Thread(target=run_fastapi)
        video_thread = Thread(target=run_video_capture)
        cleanup_thread = Thread(target=cleanup_old_alerts, daemon=True)
        fetch_and_update_cache_thread = Thread(target=fetch_and_update_cache, daemon=True)

        flask_thread.start()
        fastapi_thread.start()
        video_thread.start()
        cleanup_thread.start()
        fetch_and_update_cache_thread.start()

        # Join non-daemon threads
        flask_thread.join()
        fastapi_thread.join()
        video_thread.join()
        # Daemon threads (cleanup_thread and fetch_and_update_cache_thread) run in the background and don't need to be joined

    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        logging.info("Cleaning up resources...")
        if cap.isOpened():
            cap.release()
        client.close()
