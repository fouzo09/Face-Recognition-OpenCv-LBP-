# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:30:24 2021

@author: Fouzo
"""
import cv2
import pickle
import numpy as np
from lib.helpers import take_image, face_detector


images_path = "image path"
names       = {0: "Mafouz DIALLO", 1: "Mamadou BAH", 2 : "Thierno Amadou SOW"}
recognizer  = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("face_trainner.yml")

img  = cv2.imread(images_path)
img  = cv2.resize(img, (800, 600))

face, coords = face_detector(images_path)
(id_, conf)  = recognizer.predict(face)

if conf >= 45:
    
    name = names[id_]
    for (x, y, w, h) in coords:
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, name.upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
         0.75, (0, 255, 0), 2)

cv2.imshow(name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()