import moviepy.editor as mpy
import cv2
import numpy as np

if __name__ == '__main__':
    frames = []
    
    for i in range(92):
        frame = cv2.imread(str(i)+'.png')    
        frames.append(frame)

    clip = mpy.ImageSequenceClip(frames , fps = 25)
    clip.write_videofile("part-1.mp4", codec="libx264")