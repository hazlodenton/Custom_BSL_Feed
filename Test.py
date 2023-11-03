import mediapipe as mp
import cv2
import numpy as np
import os
import uuid


mp_drawing = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

vid = cv2.VideoCapture(0)  

with mp_hand.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:
    while(True):  
        ret, frame = vid.read()  
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        print(results)

        if results.multi_hand_landmarks:
                for num,hand in enumerate(results.multi_hand_landmarks):
                     mp_drawing.draw_landmarks(image,hand,mp_hand.HAND_CONNECTIONS)

        cv2.imshow('frame', image) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
vid.release()
cv2.destroyAllWindows()
