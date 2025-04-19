import numpy as np 
import cv2
import matplotlib.pyplot as plt
from main_function import *
from utils import *
import time

def meanShiftwindow(frame, track_window, q, weights, epsilon, max_iterations, window_size=1):
    x,y, w, h = track_window
    i = 0
    d = np.inf

    half_width = w / window_size // 2
    half_height = h / window_size // 2
    Y, X = np.mgrid[-half_width:half_width+1, -half_height:half_height+1]
    while True:
        rgb_roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)
        p = normalize_histogram(extract_histogram(hsv_roi, 8, weights=weights))
        v = np.sqrt(np.divide(q, p + 1e-5))
        weights_mat = backproject_histogram(hsv_roi, v, 8)
        pool_roi = avgpool2d(weights_mat,window_size)
        average_X_mat = X * pool_roi
        average_Y_mat = Y * pool_roi
        x_change = np.sum(average_X_mat) / np.sum(pool_roi) * window_size
        y_change = np.sum(average_Y_mat) / np.sum(pool_roi) * window_size
        x += round(x_change)
        y += round(y_change)
        i += 1
        if i >= 10:
            break
        if abs(x_change) < epsilon  and abs(y_change) < epsilon: 
            break

    return x,y, w,h


def meanShiftTracking2(cap, file_name, ROI, kernel, gamma, window_size, occlusion = False):
    ret, frame = cap.read()
    if file_name != False: 
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter(file_name, fourcc, 29, (frame.shape[1], frame.shape[0]))
    (x, y, w, h) = ROI
    x,y,w,h = adjust_size_for_centered_pooling(x,y,w,h,window_size)
    track_window = (x, y, w, h)

    roi = frame[y:y+h, x:x+w]

    weights = kernel((h, w))
    rgb_roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)
    q = normalize_histogram(extract_histogram(hsv_roi, 8, weights=weights))
    org_q = q.copy()
    histogram_list = [] 
    total_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            before = time.time()
            x,y,w,h = meanShiftwindow(frame, (x,y,w,h), q, weights, 1, 10, window_size)
            after = time.time()
            total_time += after - before
            if file_name != False: 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), frame.shape[1]//1200)
                out.write(frame)
            rgb_roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)
            new_q = normalize_histogram(extract_histogram(hsv_roi, 8, weights=weights))
            histogram_list.append(new_q.copy())
            if occlusion == False: 
                q = np.add(np.multiply((1 - gamma), org_q), np.multiply(gamma, new_q))
            else: 
                if bhattacharyya_distance(new_q, q) < occlusion: 
                    q = np.add(np.multiply((1 - gamma), org_q), np.multiply(gamma, new_q))
        else:
            break

    similarities = [bhattacharyya_distance(h, org_q) for h in histogram_list]
    if file_name != False:
        out.release()
    cap.release()
    return similarities, total_time

if __name__=="__main__":
    cap=cv2.VideoCapture("usain_bolt.mp4")
    file_name =  "./results_video/window_meanshift2.avi"
    ROI = (620, 310, 40, 40)
    gamma = 0.6
    window_size = 5
    meanShiftTracking2(cap, file_name, ROI, epanechnikov_kernel, gamma, window_size)
