import time 
import matplotlib.pyplot as plt 
import os

from meanshift import *
from meanshift2 import * 
from meanshift3 import *
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

kernel_id = 1
kernel = kernel_list[kernel_id]
gamma = 0.8
file_name = False
window_size = 15
occlusion = 0.2 

video_output = True

cap=cv2.VideoCapture(video)
if video_output: 
    file_name = "./results_video/original_meanshift.avi"
sim1, time1 = meanShiftTracking(cap, file_name, ROI, kernel, gamma, occlusion)

cap=cv2.VideoCapture(video)
if video_output: 
    file_name = "./results_video/window_meanshift2.avi"
sim2, time2 = meanShiftTracking2(cap, file_name, ROI, kernel, gamma, window_size, occlusion)

cap=cv2.VideoCapture(video)
if video_output:
    file_name = "./results_video/window_meanshift3.avi"
sim3, time3 = meanShiftTracking3(cap, file_name, ROI, kernel, gamma, window_size, occlusion)



kernel_name = ["epanechnikov_kernel", 
               "gaussian_kernel"]
kernel_name = kernel_name[kernel_id]
print("Gamma: " + str(gamma)) 
print("kernel: " + kernel_name) 
print("window_size: " + str(window_size)) 
print("video: " + video)
print("Original Meanshift: ", time1) 
print("Window Meanshift: ", time2) 
print("Naive Window Meanshift: ", time3) 
print("occlusion: " + str(occlusion))

plt.plot(sim1, label='Original Meanshift')
plt.plot(sim2, label='Window Meanshift')
plt.plot(sim3, label='Naive Window Meanshift')


# Add labels and legend
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("Gamma: " + str(gamma) + " " + kernel_name + " " + "window_size: " + str(window_size) + " " + "occlusion: " + str(occlusion) + " " +  video )
plt.legend()

# Show plot
plt.grid(True)
plt.show()
         

