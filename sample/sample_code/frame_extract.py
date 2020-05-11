import cv2
import os

def convert_to_frame(test_file):
    vidcap = cv2.VideoCapture(test_file)
    print(vidcap)
    success,image = vidcap.read()
    count = 0
    while success:
        success,image = vidcap.read()
        print("reading frame ",count,success)
        crop_img = image[:, 280:1000]
        cv2.waitKey(0)
        cv2.imwrite("frame%d.jpg" % count, crop_img)     # save frame as JPEG file
        count += 1
        
convert_to_frame("test.mp4")        