"""
Name: Başar Demir
Student Number: 150180080
"""

import scipy.io
import os
import numpy as np
import cv2
from sklearn.metrics import precision_score


# ground truth mat files path
gt_path="dataset/groundTruth/test/"

#dataset test images path
img_path = "dataset/images/test/"

#sum of all precisions
total_precision = 0

for file in os.listdir(gt_path):
    #reads mat file
    gt_file_path = gt_path + file
    mat = scipy.io.loadmat(gt_file_path)['groundTruth']
    
    #merges all boundary matrices
    gt_edges = mat[0,0][0][0][1]
    for i in range(1,len(mat[0])):
        gt_edges = np.logical_or(gt_edges, mat[0,i][0][0][1])
    
    #reads test image  
    img = cv2.imread(img_path + file.split(".")[0] + ".jpg")
    #performs edge detection
    edges = cv2.Canny(img,200,500)
    #sets edges as 1
    edges[edges>0] = 1
    
    #calculates precision and adds to total precision
    total_precision += precision_score(gt_edges, edges, average='micro')
 
print("Average Precision: " + str(total_precision/200))