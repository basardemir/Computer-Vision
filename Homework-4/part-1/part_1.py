"""
Name: Başar Demir
Student Number: 150180080
"""

import cv2
import os
import numpy as np
from numpy.linalg import pinv
import moviepy.editor as mpy

FOLDER_NAME = "../DJI_0101"
WINDOW_SIZE = 10

class LucasKanade():
    def __init__(self, prevFrame, currentFrame):
        self.originalFrame = cv2.imread(FOLDER_NAME+ "/"+ prevFrame)
        self.prevFrame = cv2.cvtColor(self.originalFrame, cv2.COLOR_BGR2GRAY)
        self.currentFrame = cv2.cvtColor(cv2.imread(FOLDER_NAME+ "/"+ currentFrame), cv2.COLOR_BGR2GRAY)
        
        self.height, self.width = self.prevFrame.shape
        
        self.V = np.zeros((self.height, self.width, 2))
        
        self.corners = []
        
        self.threshold = 0.1
    
    def getDerivative(self, row, col):
            Ix = (0.25 * self.prevFrame[row, col+1] + 0.25 * self.currentFrame[row, col+1] + 0.25 * self.prevFrame[row+1, col+1] + 0.25 * self.currentFrame[row+1, col+1]) - (0.25 * self.prevFrame[row, col] + 0.25 * self.currentFrame[row, col] + 0.25 * self.prevFrame[row+1, col] + 0.25 * self.currentFrame[row+1, col])
            Iy = (0.25 * self.prevFrame[row+1, col] + 0.25 * self.currentFrame[row+1, col] + 0.25 * self.prevFrame[row+1, col+1] + 0.25 * self.currentFrame[row+1, col+1]) - (0.25 * self.prevFrame[row, col] + 0.25 * self.currentFrame[row, col] + 0.25 * self.prevFrame[row, col+1] + 0.25 * self.currentFrame[row, col+1])
            It = (0.25 * self.currentFrame[row, col] + 0.25 * self.currentFrame[row+1, col] + 0.25 *self.currentFrame[row, col+1] + 0.25 * self.currentFrame[row+1, col+1]) - (0.25 * self.prevFrame[row, col] + 0.25 * self.prevFrame[row+1, col] + 0.25 * self.prevFrame[row, col+1] + 0.25 * self.prevFrame[row+1, col+1])
            return Ix, Iy, -It
            
    def getVelocity(self):
        self.prevFrame = cv2.GaussianBlur(self.prevFrame,(3,3),cv2.BORDER_DEFAULT)
        self.prevFrame = cv2.GaussianBlur(self.prevFrame,(3,3),cv2.BORDER_DEFAULT)
        self.corners = np.int0(cv2.goodFeaturesToTrack(self.prevFrame,10000, 0.15, 10,blockSize=5))
        
        for corner in self.corners:
            col, row = corner.flatten()
            if not (((row + WINDOW_SIZE // 2) < self.height) and ((row - WINDOW_SIZE // 2) > 0) and ((col + WINDOW_SIZE // 2) < self.width) and ((col - WINDOW_SIZE // 2) > 0)):
                continue
            
            A = np.zeros((WINDOW_SIZE**2, 2))
            b = np.zeros((WINDOW_SIZE**2, 1))
            
            counter = 0
            for box_row in range(-WINDOW_SIZE // 2 , WINDOW_SIZE // 2 ):
                for box_col in range(-WINDOW_SIZE // 2 , WINDOW_SIZE // 2 ):
                    A[counter, 0], A[counter, 1], b[counter, 0] = self.getDerivative(row+box_row, col+box_col)
                    counter+=1
                
            self.V[row, col, :] = (pinv(A.T @ A)@(A.T)@b).T

    def drawArrows(self):
        for corner in self.corners:
            col, row = corner.flatten()
            if((self.V[row,col,0]**2 + self.V[row,col,1]**2) > self.threshold):
                self.originalFrame = cv2.arrowedLine(self.originalFrame, (col, row), (col + int(100* self.V[row,col,0]), row+ int(100*self.V[row,col,1])), color = (255, 0, 0), thickness = 4)
        
    def getFrame(self):
        self.getVelocity()
        self.drawArrows()
        return self.originalFrame[:,:,::-1]
    
if __name__ == '__main__':
    frames = []
    
    prev_frame = "00000.png"

    for current_frame in os.listdir(FOLDER_NAME):
        if(current_frame == "00000.png"):
            continue
        
        LK = LucasKanade(prev_frame, current_frame)
        frames.append(LK.getFrame())
        
        prev_frame = current_frame
    
    clip = mpy.ImageSequenceClip(frames , fps = 25)
    clip.write_videofile("part-1.mp4", codec="libx264")