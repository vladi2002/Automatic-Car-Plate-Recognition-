import cv2
import numpy as np

"""
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
In this method we define a color range which serves as mask for colour segmentation
of the input image. The range used is [8, 100, 100] [40, 255, 255].

Additionally, morphological closing with kernel 15x15 rectangle
is performed on the mask to remove noise.

That way orange plates are acquired.
Outputs:(One)
	1. result: image with only the mask coloured
	type: Numpy array (imread by OpenCV package)
"""
def orange_detection(image):
    hsv_single_plate = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define color range
    colorMin = np.array([8, 100, 100])
    colorMax = np.array([40, 255, 255])
    mask = cv2.inRange(hsv_single_plate, colorMin, colorMax)

    # Plot the masked image (where only the selected color is visible)
    result = cv2.bitwise_and(hsv_single_plate, hsv_single_plate, mask=mask)
    result_RGB = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

    element = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(15, 15))
    cv2.morphologyEx(src=result,
                     op=cv2.MORPH_CLOSE,
                     kernel=element,
                     dst=result)
    return result

"""
Extracts multiple plates per one image and performs rotation and wrapping
Inputs:(One)
	1. img: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
In this method we extract the plate from the image after the colour segmentation is applied.
Steps:
    1. Binarize image after colour segmentation and find its connected components (all non-black pixels)
    2. For each connected component apply the minAreaRect function to crop it out of the image and rotate it to 
    horizontal orientation
That way orange plates are acquired.
Outputs:(One)
	1. result: cropped image with only the plate (or images of rotated orange parts of the original)
	type: Numpy array (imread by OpenCV package)
"""
def extract_multiple_plate(img):
    def extract_points(labeled_image, num_labels):
        x_temp, y_temp = np.nonzero(labeled_image)
        coords = np.column_stack((y_temp, x_temp))
        vals = labeled_image[x_temp, y_temp]
        res = {k: coords[vals == k] for k in range(1, num_labels)}
        return res

    orange_plate = orange_detection(img)

    gray = cv2.cvtColor(orange_plate, cv2.COLOR_BGR2GRAY)
    ret, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    components_info = cv2.connectedComponents(binary_image, 4, cv2.CV_32S)
    (numLabels, labels) = components_info

    if numLabels == 1:
        return None

    cropped_plates = []
    all_points = extract_points(labels, numLabels)
    for i in range(1, numLabels):
        points = all_points[i]
        if len(points) == 0:
            continue
        ((x, y), (width, height), angle) = cv2.minAreaRect(points)
        if angle > 45:
            angle = angle - 90
        (h, w) = img.shape[:2]

        # start = datetime.datetime.now()
        copy_img = np.zeros(img.shape, np.uint8)
        for point in points:
            copy_img[point[1]][point[0]] = img[point[1]][point[0]]
        # end = datetime.datetime.now()

        rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        rotated = cv2.warpAffine(copy_img, rotation_matrix, (w, h))
        rotated_original = cv2.warpAffine(img, rotation_matrix, (w, h))

        indices = np.nonzero(rotated)

        minx = indices[1].min()
        maxx = indices[1].max()
        miny = indices[0].min()
        maxy = indices[0].max()

        final_cropped = rotated_original[miny: maxy+1, minx: maxx+1]
        cropped_plates.append(final_cropped)
    return cropped_plates
