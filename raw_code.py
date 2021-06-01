#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Nikhil RA__052


# In[5]:


#convert video to frames
import cv2
import time
vidcap = cv2.VideoCapture(r'C:\Users\nikhi\Downloads\sample_video.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        #img saved in sample area
        cv2.imwrite(r"C:\Users\nikhi\Downloads\image"+str(count)+".jpg", image)     
        
        #detect car using sample frames-
        face_cascade = cv2.CascadeClassifier(r'C:\Users\nikhi\Downloads\opencv-car-detection-master\cars.xml')   #training dataset
        time.sleep(1)
        img = cv2.imread(r'C:\Users\nikhi\Downloads\image'+str(count)+'.jpg', 1)
        
        #image converted to grey scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #detecting the image 
        cars = face_cascade.detectMultiScale(gray, 1.1, 1)
        for (x, y, w, h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 
        plt.figure(figsize=(10,20))

        plt.imshow(img)
    return hasFrames

#starting of code
sec = 0
frameRate = 0.5 
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)


# In[6]:


#training dataset
face_cascade = cv2.CascadeClassifier(r'C:\Users\nikhi\Downloads\opencv-car-detection-master\cars.xml')
img = cv2.imread(r'C:\Users\nikhi\Downloads\opencv-car-detection-master\car4.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,20))

plt.imshow(gray)
cars = face_cascade.detectMultiScale(gray, 1.1, 1)
for (x, y, w, h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) 
plt.figure(figsize=(10,20))

plt.imshow(img)


# In[ ]:


for i in range(1,100):
    img=cv2.imread('comtent/'+str(i)+'.png',1)

