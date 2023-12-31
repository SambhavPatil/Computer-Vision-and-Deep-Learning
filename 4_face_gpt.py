# -*- coding: utf-8 -*-
"""4_face_gpt

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cltyYjeg253b5ohnPd4eH6VYCsgIwKX_
"""

pip install cmake
pip install dlib
pip install face_recognition

import cv2
import os

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory path containing known face images
known_images_dir = "path_to_known_images_directory"

# Dictionary to store known face encodings mapped to their names
known_encodings = {}

# Load and encode known face images from the directory
for filename in os.listdir(known_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(known_images_dir, filename)
        known_image = cv2.imread(image_path)
        gray_known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the known images
        faces = face_cascade.detectMultiScale(gray_known_image, scaleFactor=1.1, minNeighbors=5)

        # Encode faces (for simplicity, assuming only one face per known image)
        for (x, y, w, h) in faces:
            roi = gray_known_image[y:y+h, x:x+w]
            encoding = cv2.resize(roi, (100, 100))  # Resize for consistent encoding
            known_encodings[name] = encoding

# Path to the test image for face recognition
test_image_path = "path_to_test_image.jpg"

# Load the test image
test_image = cv2.imread(test_image_path)
gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the test image
faces = face_cascade.detectMultiScale(gray_test_image, scaleFactor=1.1, minNeighbors=5)

# Loop through each face found in the test image
for (x, y, w, h) in faces:
    roi = gray_test_image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (100, 100))  # Resize for consistent encoding

    # Compare the current face encoding with known face encodings
    for name, known_encoding in known_encodings.items():
        similarity = cv2.matchTemplate(roi, known_encoding, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Set a similarity threshold
        if similarity >= threshold:
            cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(test_image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            break  # Break loop if a match is found

# Display the test image with recognized faces
cv2.imshow('Test Image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()