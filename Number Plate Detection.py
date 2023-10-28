#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


np_cascade=cv2.CascadeClassifier('np.xml')


# In[3]:


cap = cv2.VideoCapture(0)


# In[ ]:


while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = np_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(300) & 0xff
    if k==27:
        break


# In[ ]:


cv2.destroyAllWindows()

cap.release()

