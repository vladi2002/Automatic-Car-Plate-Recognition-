import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
This method is used to read the sample numbers
Returns a list of the sample numbers and 
a list of the x coordinates of the rightmost point from each number. 
"""
def get_sample_numbers():
    all_numbers = []
    all_numbers_maxx = []
    for i in range(10):
        number = plt.imread('dataset/SameSizeNumbers/' + str(i) + '.bmp')
        if len(number.shape) > 2:
            new_number = np.zeros((number.shape[0], number.shape[1]))
            for i in range(number.shape[0]):
                for j in range(number.shape[1]):
                    new_number[i][j] = sum(number[i][j][0:3])
                    if new_number[i][j] != 0:
                        new_number[i][j] = 255
            new_number = new_number.astype(np.uint8)
            all_numbers.append(new_number)
        else:
            all_numbers.append(number)

        n = all_numbers[-1]
        indices = np.nonzero(n)
        maxx = max(indices[1])

        all_numbers_maxx.append(maxx)
    return all_numbers, all_numbers_maxx


"""
This method is used to read all the sample letters.
Returns a dictionary with keys the letter and values the image of the letter and 
a dictionary with keys the letters and values the x coordinate of the rightmost point from the letter.
"""
def get_sample_letters():
    all_letters = dict()
    all_letters_maxx = dict()
    directory = 'dataset/SameSizeLetters/'
    ls = os.listdir(os.curdir + '/' + directory)
    for l in ls:
        letter = plt.imread(directory + l)
        if len(letter.shape) > 2:
            new_letter = np.zeros((letter.shape[0], letter.shape[1]))
            for i in range(letter.shape[0]):
                for j in range(letter.shape[1]):
                    new_letter[i][j] = sum(letter[i][j][0:3])
                    if new_letter[i][j] != 0:
                        new_letter[i][j] = 255
            new_letter = new_letter.astype(np.uint8)
            all_letters[l[0]] = new_letter
        else:
            all_letters[l[0]] = letter

        n = all_letters[l[0]]
        indices = np.nonzero(n)

        # minx = min(indices[1])
        maxx = max(indices[1])
        # miny = min(indices[0])
        # maxy = max(indices[0])
        all_letters_maxx[l[0]] = maxx
        # print(minx, maxx, miny, maxy, l[0])
    return all_letters, all_letters_maxx


numbers, numbers_maxx = get_sample_numbers()
letters, letters_maxx = get_sample_letters()

# The scheme we use to correct mismatches from the pattern when they occur.
correcting_scheme = {'1': 'T', '2': 'Z', '3': 'R', '4': 'K', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P', '0': 'D',
                     'B': '8', 'D': '0', 'F': '5', 'G': '6', 'H': '8', 'J': '4', 'K': '4', 'L': '1', 'M': '8', 'N': '2',
                     'P': '9', 'R': '8', 'S': '5', 'T': '2', 'V': '1', 'X': '8', 'Z': '2'}

"""
This method is used to remove the shadow from the initial plate image, then resize it to a standard size and binarize.
Returns the binary image obtained after all the manipulations.
"""
def one_image_plate(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5, 5), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)

    result_norm = cv2.resize(result_norm, (400, 85))
    gray = cv2.cvtColor(result_norm, cv2.COLOR_RGB2GRAY)

    ret2, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2,2)), iterations=1)

    thresh = cv2.bitwise_not(thresh)
    return thresh


"""
This method is used to recognize a given character. First we resize the character,
then we xor it against all the samples, and we return the character which fits best.
"""
def recognize_basic(character):
    xors = []
    for i in range(10):
        normalized = cv2.resize(character, (numbers_maxx[i], 85))
        final = cv2.copyMakeBorder(
            normalized, 0, 0, 0, 100 - normalized.shape[1], cv2.BORDER_CONSTANT)
        if numbers[i].shape != final.shape:
            continue
        xor = cv2.bitwise_xor(final, numbers[i])
        c = np.count_nonzero(xor)
        xors.append((c*100/(numbers_maxx[i]*85), str(i)))

    for i in letters.keys():
        normalized = cv2.resize(character, (letters_maxx[i], 85))
        final = cv2.copyMakeBorder(
            normalized, 0, 0, 0, 100 - normalized.shape[1], cv2.BORDER_CONSTANT)
        if letters[i].shape != final.shape:
            continue
        xor = cv2.bitwise_xor(final, letters[i])
        c = np.count_nonzero(xor)
        xors.append((c*100/(letters_maxx[i]*85), i))

    xors.sort()
    return xors[0][1]


"""
This method is used to detect abnormalities in the plate we detect and try to fix them 
given the pattern for Dutch license plates. 
Returns the plate obtained after the algorithm.
Currently, this method works when there is exactly one mistake in the plate provided.
"""
def fix_pattern(s):
    if len(s) != 8:
        return ''

    groups = []
    j = 0
    while j < len(s):
        if s[j] == '-':
            j += 1
            continue
        groups.append([])
        while j < len(s) and s[j] != '-':
            groups[-1].append(s[j])
            j += 1

    if len(groups) != 3:
        return ''

    lnmap = dict()
    dif = {'l': 'n', 'n': 'l'}

    for i in range(3):
        if len(groups[i]) == 3:
            count_numbers = 0
            for j in groups[i]:
                if '0' <= j <= '9':
                    count_numbers += 1
            if count_numbers >= 2:
                lnmap[i] = 'n'
            else:
                lnmap[i] = 'l'
            for j in range(3):
                if j % 2 == i % 2:
                    lnmap[j] = lnmap[i]
                else:
                    lnmap[j] = dif[lnmap[i]]

    if 0 in lnmap.keys():
        for i in range(3):
            if lnmap[i] == 'l':
                for j in range(len(groups[i])):
                    if '0' <= groups[i][j] <= '9':
                        groups[i][j] = correcting_scheme[groups[i][j]]
            if lnmap[i] == 'n':
                for j in range(len(groups[i])):
                    if 'A' <= groups[i][j] <= 'Z':
                        groups[i][j] = correcting_scheme[groups[i][j]]
        result = ''
        for group in groups:
            for c in group:
                result += c
            result += '-'
        return result[:-1]

    not_found = -1

    for i in range(len(groups)):
        if '0' <= groups[i][0] <= '9' and '0' <= groups[i][1] <= '9':
            lnmap[i] = 'n'
        elif 'A' <= groups[i][0] <= 'Z' and 'A' <= groups[i][1] <= 'Z':
            lnmap[i] = 'l'
        else:
            not_found = i

    if len(lnmap.keys()) < 2:
        return s

    if 'n' not in lnmap.values():
        lnmap[not_found] = 'n'
    else:
        lnmap[not_found] = 'l'

    for i in range(3):
        if lnmap[i] == 'l':
            for j in range(len(groups[i])):
                if '0' <= groups[i][j] <= '9':
                    groups[i][j] = correcting_scheme[groups[i][j]]
        if lnmap[i] == 'n':
            for j in range(len(groups[i])):
                if 'A' <= groups[i][j] <= 'Z':
                    groups[i][j] = correcting_scheme[groups[i][j]]
    result = ''
    for group in groups:
        for c in group:
            result += c
        result += '-'
    return result[:-1]


""" 
This is the method called from Recognize.py that given a image of a plate returns the actual plate as a string.
"""
def recognize_plate_image(img):

    # obtain the binary image
    binary_image = one_image_plate(img)

    # the components in the binary image
    components_info = cv2.connectedComponentsWithStats(
        binary_image, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = components_info

    characters = []
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # the conditions which decide whether a certain component is a character
        if 15 < w < 60 and 30 < h < 70 and x+w != binary_image.shape[1] - 1 \
                and y+h != binary_image.shape[0] - 1 and x != 0 and y != 0:
            characters.append((x, x+w, binary_image[y:y+h, x:x+w]))

    if len(characters) < 3:
        return ''

    characters.sort()
    plate = ''

    # list which stores the distances between consecutive characters after the sorting.
    # The biggest two distances will indicate a dash in the plate.
    distances = []
    for i in range(len(characters)-1):
        distances.append(characters[i+1][0] - characters[i][1])

    distances.sort()

    a = distances[-1]
    b = distances[-2]

    for i in range(len(characters)-1):
        # append the recognized character
        plate += recognize_basic(characters[i][2])

        # put dash if the distance is one of the two largest ones.
        if characters[i+1][0] - characters[i][1] == a or characters[i+1][0] - characters[i][1] == b:
            plate += '-'

    plate += recognize_basic(characters[len(characters)-1][2])

    # we call the function which will try to fix the plate if there is any mismatch according to the pattern.
    plate = fix_pattern(plate)
    return plate


