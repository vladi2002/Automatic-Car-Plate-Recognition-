#!/bin/bash

T=$(find . -type f -wholename "./dataset/trainingsvideo.avi")
F=$(find . -type f -wholename "./Output.csv")

python main.py --file_path $T --output_path $F

G=$(find . -type f -wholename "./dataset/groundTruth.csv")

python evaluation.py --file_path $F  --ground_truth_path $G

read
