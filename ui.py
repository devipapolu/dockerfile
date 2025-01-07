import os
import streamlit as st
import requests
import base64
import cv2
import numpy as np
import pandas as pd
from pymongo import MongoClient
from PIL import Image
from datetime import datetime

# MongoDB setup
client = MongoClient("mongodb+srv://sahasra:4Jan%401998@visitormanagement.th43u.mongodb.net/?retryWrites=true&w=majority&appName=VisitorManagement")
db = client["visitor_management"]
faceCollection = db["employees"]

# Directory to save registered images
image_save_directory = "registered_images"
os.makedirs(image_save_directory, exist_ok=True)

# Convert image to base64
def to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Save image locally
def save_image(person_id, image):
    file_path = os.path.join(image_save_directory, f"{person_id}.jpg")
    cv2.imwrite(file_path, image)
    return file_path

# Register user function with additional fields
def register_user(person_id, name, gender, captured_images):
    try:
        if not captured_images:
            st.error("No images provided for registration.")
            return

        # Use the first captured image for saving and encoding
        main_image = captured_images[0]
        image_path = save_image(person_id, main_image)
        image_base64 = to_base64(main_image)

        # Create 512-D embeddings for all images
        all_embeddings = []
        for image in captured_images:
            base64_image = to_base64(image)
            response = requests.post("http://localhost:8000/getFaceEmbeddings", json={'base64_image': base64_image})
            response.raise_for_status()
            face_data = response.json()

            if 'face_embeddings' in face_data and face_data['face_embeddings']:
                embedding = face_data['face_embeddings'][0]
                all_embeddings.append({"angle": "unspecified", "embedding": embedding})
            else:
                st.warning("No face detected in one of the images. Skipping.")

        if all_embeddings:
            # Check if the person already exists in the collection
            existing_person = faceCollection.find_one({"person_id": person_id})

            if existing_person:
                # Update the existing person's embeddings
                faceCollection.update_one(
                    {"person_id": person_id},
                    {"$push": {"embeddings": {"$each": all_embeddings}}}
                )
                st.success(f"Embeddings added for existing user: {name}")
            else:
                # Insert a new person document with additional fields
                faceCollection.insert_one({
                    "person_id": person_id,
                    "name": name,
                    "gender": gender,
                    "embeddings": all_embeddings,
                    "base64_image": image_base64,  # Store base64 image
                    "image_url": image_path,       # Path to saved image
                    "registered_on": datetime.utcnow()  # Registration timestamp
                })
                st.success("User registered successfully")
        else:
            st.error("No valid face embeddings were detected in the captured images.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the face embedding API: {e}")
    except ValueError:
        st.error("Invalid response from face embedding API. Please check the server.")

# Streamlit UI
st.title("Register New User")

# Collect user details
person_id = st.text_input("Enter a unique ID for the person:")
name = st.text_input("Enter the person's name:")
gender = st.selectbox("Select gender:", ["Male", "Female", "Other"])

photo_option = st.radio('Upload image from', ['Folder', 'Webcam'], horizontal=True)

# Initialize captured images list
if "captured_images" not in st.session_state:
    st.session_state["captured_images"] = []

# Initialize previous photo state
if "last_photo" not in st.session_state:
    st.session_state["last_photo"] = None

# Image capture from webcam
if photo_option == 'Webcam':
    photo = st.camera_input("Capture a face image")
    if photo:
        img = Image.open(photo)
        image = np.array(img)

        # Check if the current image is different from the last one
        if not np.array_equal(st.session_state["last_photo"], image):
            st.session_state["captured_images"].append(image)
            st.session_state["last_photo"] = image

# Image upload from folder
elif photo_option == 'Folder':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        image = np.array(img)
        st.session_state["captured_images"].append(image)

# Display captured images
st.subheader("Captured Images")
for idx, img in enumerate(st.session_state["captured_images"]):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(img, caption=f"Image {idx + 1}", use_container_width=True)
    with col2:
        if st.button(f"Remove Image {idx + 1}", key=f"remove_{idx}"):
            st.session_state["captured_images"].pop(idx)
            st.experimental_rerun()

# Register button
if st.button("Register"):
    if person_id and name and gender and st.session_state["captured_images"]:
        register_user(person_id, name, gender, st.session_state["captured_images"])
        # Clear captured images after registration
        st.session_state["captured_images"] = []
        st.session_state["last_photo"] = None
    else:
        st.error("Please fill in all the details and capture/upload valid images.")


