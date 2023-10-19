import cv2
import numpy as np

video_output_path = "output_video.mp4" 
image1 = cv2.imread("jems.png")
image2 = cv2.imread("erum.png")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_output_path, fourcc, 10, (1280, 720))
w1 = 0.0
w2 = 1.0
weight_array = np.full((720, 1280), w1, dtype=np.float32)
steps = 0.05 

image_height, image_width = 100, 100  
x_start = 100  
y_start = 100  
x_end = x_start + image_width
y_end = y_start + image_height
weight_array[y_start:y_end, x_start:x_end] = w2
np.save('weight_array.npy', weight_array)
numframes = 200 
isW1 = True
isW2 = False
for i in range(numframes) :
    resized_image1 = cv2.resize(image1, (weight_array.shape[1], weight_array.shape[0]))
    resized_image2 = cv2.resize(image2, (weight_array.shape[1], weight_array.shape[0]))
    result = cv2.addWeighted(resized_image1,w1,resized_image2,w2,0)
    if w1 >= 1 and w2 <= 0 :
        isW2 = True
        isW1 = False
    elif w1 <= 0 and w2 >= 1 :
        isW2 = False
        isW1 = True
    if isW1 :
        w1 += steps
        w2 -= steps
    elif isW2 :
        w1 -= steps
        w2 += steps
    video_writer.write(result)
video_writer.release()





