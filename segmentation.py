import cv2
import skimage.filters
import numpy as np

def segmentImage(image):

    # maybe need some preprocessing to rotate the image to be oriented horizontally... 

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.threshold(grayscale, 128, 192, cv2.THRESH_OTSU)
    blurred = skimage.filters.gaussian(binarized)

    # find histograms of horizontal lines, and if above certain threshold, determine upper/lower bounds of characters
    vertical_threshold = 0.5
    row_histograms = np.sum(image,axis=1)
    top_border, bottom_border = np.nonzero(row_histograms > vertical_threshold)[0][[0,-1]]


    # find histograms of vertical lines, if above threshold you find regions separating individual characters. 

    horizontal_threshold = 0.5
    column_histograms = np.sum(image, axis=0)
    vertical_boundaries = np.nonzero(column_histograms > horizontal_threshold)[0]
    # separate regions into individual sections and return. 

    subImages = []
    for i in range(len(vertical_boundaries) - 1):
        subImage = image[vertical_boundaries[i]:vertical_boundaries[i+1], top_border: bottom_border]
        subImages.append(subImage)
    return subImages
