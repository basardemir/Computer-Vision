import pyautogui
import time
import cv2
import numpy as np
import imutils


def pushButton(approx):
    if approx == 3:
        pyautogui.press("A")
    elif approx == 4:
        pyautogui.press("S")
    elif approx == 6:
        pyautogui.press("F")
    elif approx == 10:
        pyautogui.press("D")

if __name__ == '__main__':
    time.sleep(3)

    while True:
        ss = pyautogui.screenshot()
        gray = cv2.cvtColor(np.array(ss), cv2.COLOR_BGR2GRAY)
        gray = gray[800:, 840:1200]
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
        
        zero_matrix = thresh[: , [0, 1, 2, 357, 358, 359]]
        
        if np.all(zero_matrix < 255):
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            
            if(len(contours)==0):
                continue
            
            approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
            pushButton(len(approx))


