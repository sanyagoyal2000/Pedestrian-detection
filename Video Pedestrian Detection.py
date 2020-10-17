#!/usr/bin/env python
# coding: utf-8

# ### Pedistrian Detection

# In[2]:


import cv2
import numpy as np


# In[8]:


# Create our body classifier
body_classifier = cv2.CascadeClassifier('/Users/sanyagoyal/Documents/Python Programs/haarcascades/cascades/haarcascade_pedestrian.xml')


# In[9]:


# Initiate video capture for video file
cap = cv2.VideoCapture('image_examples/walking.avi')


# In[10]:


# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
