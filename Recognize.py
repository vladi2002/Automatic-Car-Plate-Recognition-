import numpy as np
from matplotlib import pyplot as plt
from recognize_plate import recognize_plate_image

"""
In this file, we define our own segment_and_recognize function.
Inputs:(One)
	1. plate_imgs: array of arrays of cropped images corresponding to one frame
Steps:
    1. Segment the plates character by character
    2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
    3. Recognize the character by comparing the distances
    4. Apply logic to correct number
    5. Record number if it is valid. To each image corresponds one number at most
Outputs:(One)
    1. result: recognized plate numbers
    type: list of lists, each element in result is a string. Each list of strings corresponds to one frame
"""
def segment_and_recognize(plate_imgs):
    result = []
    for plate in plate_imgs:
        if plate is None:
            result.append("")
        else:
            if type(plate) == np.ndarray:
                recognized_number = recognize_plate_image(plate)
                result.append(recognized_number)
            else:
                manyPlates = []
                for actual_plate in plate:
                    recognized_number = recognize_plate_image(actual_plate)
                    if recognized_number != "":
                        manyPlates.append(recognized_number)
                result.append(manyPlates)
    return result
    