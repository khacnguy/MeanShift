import time 
import matplotlib.pyplot as plt 
import os

from meanshift import *
from meanshift2 import * 
from meanshift3 import *
from meanshift4 import *
from main_function import *
from utils import *

file_dir_org = "./results_video/" 
vid_list = ["usain_bolt.mp4", 
            "resized_video.mp4"]
ROI_list = [(620, 310, 40, 40),
            (3100, 1550, 200, 200)]
kernel_list = [epanechnikov_kernel, 
               gaussian_kernel]

vid_id = 1
video = vid_list[vid_id]    
ROI = ROI_list[vid_id]

kernel_id = 0
kernel = kernel_list[kernel_id]
gamma = 0.8
file_name = False
occlusion = 0.2 

video_output = True

cap=cv2.VideoCapture(video)
if video_output: 
    file_name = "./results_video/original_meanshift.avi"
sim1, time1 = meanShiftTracking(cap, file_name, ROI, kernel, gamma, occlusion)


cap=cv2.VideoCapture("resized_video.mp4")
if video_output: 
    file_name =  "./results_video/original_meanshift_bpw.avi"
sim2, time2 = meanShiftTrackingBPW(cap, file_name, ROI, kernel, gamma, occlusion)


kernel_name = ["epanechnikov_kernel", 
               "gaussian_kernel"]
kernel_name = kernel_name[kernel_id]
print("Gamma: " + str(gamma)) 
print("kernel: " + kernel_name) 
print("video: " + video)
print("Original Meanshift: ", time1) 
print("KW Meanshift: ", time2) 
print("occlusion: " + str(occlusion))

plt.plot(sim1, label='Original Meanshift')
plt.plot(sim2, label='KW Meanshift')


plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("Gamma: " + str(gamma) + " " + kernel_name + " " + "occlusion: " + str(occlusion) + " " +  video )
plt.legend()

# Show plot
plt.grid(True)
plt.show()
         

