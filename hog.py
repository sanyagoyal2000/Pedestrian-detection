#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 
import imutils 
import numpy as np 
# Initializing the HOG person 
# detector 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   
# Reading the Image 
image = cv2.imread('crosswalk.jpg') 
#print(np.shape(image))
   
# Resizing the Image 
image = imutils.resize(image, 
                       width=min(450, image.shape[1])) 
   
# Detecting all the regions in the  
# Image that has a pedestrians inside it 
(regions, _) = hog.detectMultiScale(image,  
                                    winStride=(1, 1), 
                                    padding=(1, 1), 
                                    scale=1.05) 
i=0   
# Drawing the regions in the Image 
for (x, y, w, h) in regions: 
    
    if i!=1:
        x=x+10
        cv2.rectangle(image, (x, y),  
                  (x + w-24, y + h),  
                  (0, 0, 255), 2)
        
    else:
        x=x+36
        cv2.rectangle(image, (x, y),  
                  (x + w-34, y + h),  
                  (0, 0, 255), 2)
        
  
    i+=1
  
cv2.imshow("Image", image) 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:
