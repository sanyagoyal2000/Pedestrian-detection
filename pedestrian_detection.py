#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt


# In[2]:


img = cv2.imread('crosswalk.jpg',1)


# In[3]:


cascade = cv2.CascadeClassifier('haarcascades/cascades/haarcascade_pedestrian.xml')


# In[4]:


gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[5]:


result = cascade.detectMultiScale(gray_img)


# In[6]:


print('Pedestrians found:',len(result))


# In[7]:


for x,y,w,h in result:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# In[10]:


plt.imshow(img)


# In[ ]:



