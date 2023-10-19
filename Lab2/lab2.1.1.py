import cv2
import numpy as np
from matplotlib import pyplot as plt
video_output_path = "output_video.mp4" 
image1 = cv2.imread("img/jems.png")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_output_path, fourcc, 5, (1280, 720))
image1 = image1.astype(float)
#for g(x,y) = a * f(x,y) + b 
image1 = cv2.resize(image1,(1280,720))
init_a_factor = 1
init_b_factor = 0

a_factor = init_a_factor
b_factor = init_b_factor

steps_a = 0.01
steps_b = 5

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 0)  # White color
thickness = 2
position = (100, 100) 

for i in range(50) :
    result = a_factor * image1 + b_factor
    result = np.clip(result,0,255).astype(np.uint8)
    cv2.putText(result, "fix a, plus b", position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    video_writer.write(result)
    b_factor += steps_b

a_factor = init_a_factor
b_factor = init_b_factor
for i in range(50) :
    result = a_factor * image1 + b_factor
    result = np.clip(result,0,255).astype(np.uint8)
    cv2.putText(result, "fix a, minus b", position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    video_writer.write(result)
    b_factor -= steps_b

a_factor = init_a_factor
b_factor = init_b_factor
for i in range(50) :
    result = a_factor * image1 + b_factor
    result = np.clip(result,0,255).astype(np.uint8)
    cv2.putText(result, "fix b, plus a", position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    video_writer.write(result)
    a_factor += steps_a

a_factor = init_a_factor
b_factor = init_b_factor
for i in range(50) :
    result = a_factor * image1 + b_factor
    result = np.clip(result,0,255).astype(np.uint8)
    cv2.putText(result, "fix b, minus a", position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    video_writer.write(result)
    a_factor -= steps_a

video_writer.release()