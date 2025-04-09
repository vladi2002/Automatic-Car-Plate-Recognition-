# License plate recognition system

## About
This project was done for the course Image Processing CSE2225 at TU Delft by:
- Vladimir Petkov
- Dido Dimitrov

The aim was to create a system for recognizing license plates. 
The application adheres as close as possible to the requirements specified by the course staff.

## Running the system

The script to run both the localization, recognition, and evaluation can be found 
in the evaluator.sh file and can be executed by using the terminal in the project 
directory and executing the command `./evaluator.sh`

As arguments, you can provide:
- `T` - the filepath to the video(avi format recommended) that you want to process. 
- `F` - the filepath and name of the file where you want your recognized plates and corresponding frames and timestamps to be stored (csv format recommended). 
- `G` - the filepath of the csv file containing the ground truth the evaluator will compare the generated output against.

To follow through the whole processing pipeline, you can make use of main.py. To this end, you may need only two arguments for it - the filepath to the video and the
sampling frequency. Using the dataset folder or the provided default arguments, you may see how the program works for the files applied.

***Note: you can change the sampling frequency, which affects the processing speed in the main.py file (get_args function).***

## Requirements 
All the necessary libraries are included in the requirements.txt file.

## Work Distribution
Information on the authors' work distribution can be found in the authorship_list.txt file.
