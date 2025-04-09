import datetime
import csv
import cv2
import os
import numpy as np
import pandas as pd
from Localization import extract_multiple_plate
from Recognize import segment_and_recognize
from difflib import SequenceMatcher
from detect_scene import get_scenes

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
    1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
    2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
    3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
    1. file_path: video path
    2. sample_frequency: second
    3. save_path: final .csv file path
Output: None
"""
def CaptureFrame_Process(file_path, sample_frequency, save_path):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    get_scenes(file_path, os.getcwd() + '/extracted_scenes.csv')

    ground_truth = pd.read_csv(os.getcwd() + '/extracted_scenes.csv')
    firstFrames = np.unique(ground_truth['First frame'].tolist())
    lastFrames = np.unique(ground_truth['Last frame'].tolist())
    timestamps = np.unique(ground_truth['Timestamp'].tolist())

    count_frame = 0
    extracted_plates = []
    corresponding_frame_numbers = []
    corresponding_timestamps = []
    frame_number = 0


    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        count_frame += 1
        if ret and count_frame % sample_frequency == 0:
            frame_number += sample_frequency
            # localize the plates from the image
            extracted_multiple = extract_multiple_plate(frame)
            extracted_plates.append(extracted_multiple)
            corresponding_frame_numbers.append(frame_number)
            corresponding_timestamps.append(frame_number/fps)
        # Break the loop
        elif not ret:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    # recognize the localized plates.
    recognized_plates = segment_and_recognize(extracted_plates)

    lastIndex = 0
    header = ['License plate', 'Frame no.', 'Timestamp(seconds)']

    with open(save_path, 'w', encoding='UTF8', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)

        for i in range(len(firstFrames)):
            firstFrame = firstFrames[i]
            lastFrame = lastFrames[i]
            timestamp = timestamps[i]

            foundNumbers_and_timestamps = []
            for j in range(lastIndex, len(recognized_plates)):
                frame_number = corresponding_frame_numbers[j]
                if firstFrame < frame_number <= lastFrame and recognized_plates[j] != "":
                    foundNumbers_and_timestamps.append((recognized_plates[j], corresponding_timestamps[j]))
                elif frame_number > lastFrame:
                    lastIndex = j
                    break

            foundRecognitions = []
            for pair in foundNumbers_and_timestamps:
                for number in pair[0]:
                    if number != "" and len(number) == 8:
                        foundRecognitions.append((number, pair[1]))
            if len(foundRecognitions) == 0:
                data = ["", int(timestamp*fps), timestamp]
                writer.writerow(data)
            else:
                output_tuples = make_bins(foundRecognitions)
                for t in output_tuples:
                    data = [t[0], int(t[1]*fps), t[1]]
                    writer.writerow(data)


"""
Helps with the majority voting.
"""
def make_bins(l):
    bins = []
    for t in l:
        found_d_one = False
        for b in bins:
            for compared_tuple in b:
                if SequenceMatcher(None, t[0], compared_tuple[0]).ratio() >= 0.7:
                    b.append(t)
                    found_d_one = True
                    break
                break
        if not found_d_one:
            bins.append([t])
    output_tuples = []
    for bin in bins:
        timestamp = min(bin, key=lambda x: x[1])[1]
        dic = dict()
        for tuples in bin:
            if tuples[0] not in dic.keys():
                dic[tuples[0]] = 0
            dic[tuples[0]] += 1
        finalPlates = dict(sorted(dic.items(), key=lambda item: -item[1]))
        number = list(finalPlates.keys())[0]
        output_tuples.append((number, timestamp))
    return output_tuples


start_time = datetime.datetime.now()
CaptureFrame_Process('dataset/trainingsvideo.avi', 4, 'dataset/extractedOutputTrainingsVideo.csv')
end_time = datetime.datetime.now()
print('all', end_time - start_time)
