import numpy as np
import requests
from PIL import Image
import face_recognition
from io import BytesIO
import os
import pickle
import sys  # Ensure sys module is imported


# File to store known faces and their IDs
KNOWN_FACES_FILE = "known_faces.pkl"

# Load known faces from file if it exists, else initialize an empty dictionary
if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, 'rb') as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

# Function to save known faces to file
def save_known_faces():
    with open(KNOWN_FACES_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

# Function to generate or retrieve a unique ID based on the face and associate it with a name if provided
def get_or_generate_face_id(face_encodings, name=None):
    for face_id, (known_encodings, known_name) in known_faces.items():
        if isinstance(known_encodings, np.ndarray):
            matches = face_recognition.compare_faces([known_encodings], face_encodings)
            if any(matches):
                if name is None:
                    return face_id, known_name
                else:
                    return face_id, name

    new_face_id = len(known_faces) + 1
    known_faces[new_face_id] = [face_encodings, name] if name else [face_encodings, None]
    save_known_faces()
    return new_face_id, name


# Function to detect faces and generate face IDs
def detect_and_assign_id(image_url, name=None):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img)

    print("Image Array Shape:", img_array.shape)  # Debug: Print image array shape

    face_locations = face_recognition.face_locations(img_array)

    print("Face Locations:", face_locations)  # Debug: Print face locations

    if len(face_locations) == 0:
        return None, None

    face_encodings = face_recognition.face_encodings(img_array, known_face_locations=face_locations)

    if len(face_encodings) == 0:
        return None, None

    if name is not None:
        face_id, associated_name = get_or_generate_face_id(face_encodings[0], name)
    else:
        face_id, associated_name = get_or_generate_face_id(face_encodings[0])

    return face_id, associated_name