import os
import cv2
import zipfile
import numpy as np
from PIL import Image

path = "/content/drive/MyDrive/data"

def get_image_data():
    paths = [os.path.join("/content/drive/MyDrive/data", f) for f in os.listdir(path="/content/drive/MyDrive/data")]
    faces = []
    ids = []
    
    for path in paths:
        image = Image.open(path).convert('L')
        image_np = np.array(image, 'uint8')
        id = int(os.path.split(path)[1].split(".")[0].replace("subject", " "))
        ids.append(id)
        faces.append(image_np)
    
    return np.array(ids), faces

ids, faces = get_image_data()

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)

# Below line will store the histograms for each one of the images
lbph_classifier.write('lbph_classifier.yml')

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read("/content/lbph_classifier.yml")

# Now we will check the performance of the model
test_image = "/content/drive/MyDrive/data/subject01.glasses"
image = Image.open(test_image).convert('L')
image_np = np.array(image,'uint8')

# Before giving the image to the model let's check it first
cv2.imshow(image_np)

predictions = lbph_face_classifier.predict(image_np)
print(predictions)

expected_output = int(os.path.split(test_image)[1].split('.')[0].replace("subject"," "))
print(expected_output)
