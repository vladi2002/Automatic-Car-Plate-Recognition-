import argparse

from Localization import orange_detection
import cv2
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='dataset/dummytestvideo.avi')
    args = parser.parse_args()
    return args


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
args = get_args()
cap = cv2.VideoCapture(args.file_path)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # dummy arguments for sample frequency and save_path should be changed
        detections = orange_detection(frame)
        # Display the resulting frame
        cv2.imshow('Frame', detections)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
