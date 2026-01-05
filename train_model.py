import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []

label_map = {}
current_label = 0

for person in os.listdir("dataset"):
    label_map[current_label] = person
    person_path = f"dataset/{person}"

    for img in os.listdir(person_path):
        img_path = f"{person_path}/{img}"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(current_label)

    current_label += 1

recognizer.train(faces, np.array(labels))
os.makedirs("trainer", exist_ok=True)
recognizer.save("trainer/model.yml")

print("Model trained successfully!")
