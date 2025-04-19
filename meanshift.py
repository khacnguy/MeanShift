import numpy as np 
import cv2
import matplotlib.pyplot as plt
from main_function import *
from utils import *
import time 

def meanShift(frame, track_window, q, weights, epsilon, max_iterations):
    x,y, w, h = track_window
    i = 0
    d = np.inf

    #create kernels
    half_width = w // 2
    half_height = h // 2
    Y, X = np.mgrid[-half_width:half_width+1, -half_height:half_height+1]

    while True:
        rgb_roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)
        p = normalize_histogram(extract_histogram(hsv_roi, 8, weights=weights))
        v = np.sqrt(np.divide(q, p + 1e-5))
        weights_mat = backproject_histogram(hsv_roi, v, 8)
        average_X_mat = X * weights_mat 
        average_Y_mat = Y * weights_mat
        x_change = np.sum(average_X_mat) / np.sum(weights_mat) 
        y_change = np.sum(average_Y_mat) / np.sum(weights_mat) 
        x += round(x_change)
        y += round(y_change)
        #print(x,y, x_change, y_change, i, epsilon)
        i += 1
        if i >= 10:
            break
        if abs(x_change) < epsilon and abs(y_change) < epsilon: 
            break

    return x,y, w,h
def meanShiftTracking(cap, file_name, ROI, kernel, gamma, occlusion = False):
    ret, frame = cap.read()
    if file_name != False: 
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(file_name, fourcc, 29, (frame.shape[1], frame.shape[0]))

    (x, y, w, h) = ROI
    if w % 2 == 0: w-=1 
    if h % 2 == 0: h-=1 
    track_window = (x, y, w, h)

    roi = frame[y:y+h, x:x+w]

    weights = kernel((h, w))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_roi = hsv[y:y+h, x:x+w]
    q = normalize_histogram(extract_histogram(hsv_roi, 8, weights=weights))
    org_q = q.copy()
    histogram_list = []
    total_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            before = time.time()
            x,y,w,h = meanShift(frame, (x,y,w,h), q, weights, 1, 10)
            after = time.time() 
            total_time += after - before
            if file_name != False: 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), frame.shape[1]//1200)
                out.write(frame)
            rgb_roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)
            new_q = normalize_histogram(extract_histogram(hsv_roi, 8, weights=weights))
            if occlusion == False:
                q = np.add(np.multiply((1 - gamma), org_q), np.multiply(gamma, new_q))
            else: 
                if bhattacharyya_distance(new_q, q) < occlusion:
                    q = np.add(np.multiply((1 - gamma), org_q), np.multiply(gamma, new_q))
            histogram_list.append(new_q.copy())
        else:
            break
    
    similarities = [bhattacharyya_distance(h, org_q) for h in histogram_list]
    if file_name != False: 
        out.release()
    cap.release()
    return similarities, total_time

if __name__=="__main__":
    cap=cv2.VideoCapture("resized_video.mp4")
    file_name =  "./results_video/original_meanshift.avi"
    ROI = (3100, 1550, 200, 200)
    gamma = 0.6
    occlusion = 0.2
    meanShiftTracking(cap, file_name, ROI, gaussian_kernel, gamma, occlusion)
