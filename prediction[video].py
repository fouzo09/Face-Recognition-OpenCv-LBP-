# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:37:41 2021

@author: Fouzo
"""


import cv2
import requests
import numpy as np
from lib.helpers import detect_face_on_camera

url    = "http://192.168.10.4:8080/shot.jpg"
names  = {0: "Mafouz DIALLO", 1: "Mamadou BAH", 2 : "Thierno Amadou SOW"}

#Initialize and load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trainner.yml")

while True:
    
    #Get video stream 
    img_resp = requests.get(url)
    img_arr  = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img      = cv2.imdecode(img_arr, -1)
    
    #Resize stream frame
    img      = cv2.resize(img, (800, 600)) 
    
    #Convert stream to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #Detect face 
    face_coords = detect_face_on_camera(img)
    
    
    if len(face_coords) > 0:
        for (x, y, w, h) in face_coords:
        
            face        = img_gray[y:y+h, x:x+w]
            (id_, conf) = recognizer.predict(face)
        
            if conf >= 45:
                name = names[id_]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, name.upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                 0.75, (0, 255, 0), 2)
    cv2.imshow("CAM []", img)
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()