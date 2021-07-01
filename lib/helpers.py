# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:20:40 2021

@author: Fouzo
"""

import cv2
import os
from imutils import paths
import requests
import numpy as np
import matplotlib.pyplot as plt

url   = "http://192.168.10.4:8080/shot.jpg"
HAAR_FILE  = "lib/haarcascades/haarcascade_frontalface_default.xml"

def take_image():

    NAME_OF_PERSON = input("[INFO] Entrer le nom de la personne...")
    count = 1
    
    while True:
        
        img_resp = requests.get(url)
        img_arr  = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img      = cv2.imdecode(img_arr, -1)
        img      = cv2.resize(img, (800, 600))
        
        img_to_save = cv2.resize(img, (800, 600))
        
        if len(detect_face_on_camera(img_to_save)) > 0 :
            
            detect_face_on_camera(img_to_save)
            cv2.imwrite('datasets/'+str(NAME_OF_PERSON.lower())+str(count)+'.jpg', img_to_save)
            count = count + 1
        
        cv2.imshow("CAM", img)
        
        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()

def detect_face_on_camera(image):
    
    IMG_GRAY      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    FACE_CASACADE = cv2.CascadeClassifier(HAAR_FILE)
    
    FACES         = FACE_CASACADE.detectMultiScale(IMG_GRAY,
                                                scaleFactor=1.2, 
                                                minNeighbors=4, 
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    return FACES

def face_detector(imagePath):
    
    image = cv2.imread(imagePath)
    face  = []
    
    IMG_GRAY      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    FACE_CASACADE = cv2.CascadeClassifier(HAAR_FILE)
    
    FACES         = FACE_CASACADE.detectMultiScale(IMG_GRAY,
                                                scaleFactor=1.2, 
                                                minNeighbors=4, 
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in FACES:
        roi  = IMG_GRAY[y:y+h, x:x+w]
        face = roi
        
        
    return face, FACES

def faces_detector(datasets):
    
    images = []
    labels = []
    
    imagePaths = list(paths.list_images(datasets))
    for (i, imagePath) in enumerate(imagePaths):
        
        name  = imagePath.split(os.path.sep)[-2]    
        image = cv2.imread(imagePath)
        
        IMG_GRAY      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
        FACE_CASACADE = cv2.CascadeClassifier(HAAR_FILE)
        
        FACES         = FACE_CASACADE.detectMultiScale(IMG_GRAY,
                                                    scaleFactor=1.2, 
                                                    minNeighbors=4, 
                                                    minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in FACES:
            roi = IMG_GRAY[y:y+h, x:x+w]
            images.append(roi)
            labels.append(name)
        
        
    return (labels, images)
    