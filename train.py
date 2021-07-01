# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:30:24 2021

@author: Fouzo
"""
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lib.helpers import take_image, faces_detector
import cv2
import time
import pickle
import numpy as np


names  = {0: "Mafouz DIALLO", 1: "Mamadou BAH", 2 : "Thierno Amadou SOW"}
dataset_path = "datasets"

# load the GUB faces dataset
print("[INFO] loading dataset...")
(labels, faces) = faces_detector(dataset_path)
print("[INFO] {} images in dataset".format(len(faces)))

#Encode the string labels to integer
print("[INFO] Encoding the string labels to integer")
encoder = LabelEncoder()
labels  = encoder.fit_transform(labels)

# construct our training and testing split
(trainX, testX, trainY, testY) = train_test_split(faces,
	labels, test_size=0.25, stratify=labels, random_state=42)

# train our LBP face recognizer
print("[INFO] training face recognizer...")
recognizer = cv2.face.LBPHFaceRecognizer_create(
	radius=2, neighbors=16, grid_x=8, grid_y=8)
start = time.time()
recognizer.train(testX, testY)
end = time.time()
print("[INFO] training took {:.4f} seconds".format(end - start))

#Save the model
recognizer.save("face_trainner.yml")


