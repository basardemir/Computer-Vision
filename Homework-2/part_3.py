import pyautogui
import time
from PIL import Image
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
    print("app:"+str(approx))

if __name__ == '__main__':
    time.sleep(5)
    
    while True:
        #ss = Image.open('test2.jpg')
        ss = pyautogui.screenshot()
        gray = cv2.cvtColor(np.array(ss), cv2.COLOR_BGR2GRAY)
        gray = gray[800:, 840:1200]
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
        zero_matrix = gray[: , [358,359]]
        if np.all(zero_matrix < 255):
            print("a")
            contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = imutils.grab_contours(contours)[0]
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            pushButton(len(approx))