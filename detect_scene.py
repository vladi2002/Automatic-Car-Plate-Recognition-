from scenedetect import detect, ContentDetector
import csv
import cv2

"""
Extracts list of scene transitions from video to an output file
Inputs:(One)
	1. file_path: file path of video to process
    2. output: file path to location where we will store the information about scene transitions 
Steps:
    1. Use SceneDetect's detect method to get a list of all transitions (library usage was approved by the course staff
    for this group due to working implemented version without the library)
    2. Write the data into the output csv file
That way jump cuts in the video are correctly detected as well as their corresponding frames
"""
def get_scenes(file_path, output):
    scene_list = detect(file_path, ContentDetector())
    header = ['Timestamp', 'In beeld (tot)', 'First frame', 'Last frame']
    with open(output, 'w', encoding='UTF8', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write a row to the csv file
            writer.writerow(header)

            if len(scene_list) == 0:
                cap = cv2.VideoCapture(file_path)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(cap.get(cv2.CAP_PROP_FPS))
                writer.writerow([0, num_frames / cap.get(cv2.CAP_PROP_FPS), 0, num_frames-1])

            for i, scene in enumerate(scene_list):
                begin_s = float(scene[0].get_timecode().split(":")[-1])
                begin_m = float(scene[0].get_timecode().split(":")[-2]) * 60
                begin_h = float(scene[0].get_timecode().split(":")[-3]) * 3600
                begin = begin_s + begin_m + begin_h
                if begin.is_integer():
                    begin = int(begin)
                end_s = float(scene[1].get_timecode().split(":")[-1])
                end_m = float(scene[1].get_timecode().split(":")[-2]) * 60
                end_h = float(scene[1].get_timecode().split(":")[-3]) * 3600
                end = end_s + end_m + end_h
                if end.is_integer():
                    end = int(end)
                writer.writerow([begin, end, scene[0].get_frames(), scene[1].get_frames() - 1])
