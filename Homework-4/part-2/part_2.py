"""
Name: Başar Demir
Student Number: 150180080
"""

import cv2
import os
import numpy as np
import moviepy.editor as mpy

FOLDER_NAME = "../DJI_0101"
BACKGROUND_IMAGE= "/00459.png"
THRESHOLD = 39

if __name__ == '__main__':

    backgroundFrame = cv2.cvtColor(cv2.imread(FOLDER_NAME + BACKGROUND_IMAGE), cv2.COLOR_BGR2GRAY)
    
    for i in range(20):
        backgroundFrame = cv2.GaussianBlur(backgroundFrame,(5,5),cv2.BORDER_DEFAULT)

    frames = []
    
    for current in os.listdir(FOLDER_NAME):
        currentFrame = cv2.imread(FOLDER_NAME + "/" + current)
        currentFrameBlur = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
                
        currentFrameBlur = cv2.GaussianBlur(currentFrameBlur,(5,5),cv2.BORDER_DEFAULT)

        res = cv2.subtract(backgroundFrame, currentFrameBlur)
  
        currentFrame[THRESHOLD>res] = 0
        currentFrame[THRESHOLD<=res] = 255
        
        currentFrame = cv2.dilate(currentFrame, np.ones((3,3)))
        
        frames.append(currentFrame)
        
    clip = mpy.ImageSequenceClip(frames , fps = 25)
    clip.write_videofile("part-2.mp4", codec="libx264")