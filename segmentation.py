import cv2
import skimage.filters


def segmentImage(image):

    # maybe need some preprocessing to rotate the image to be oriented horizontally... 

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.threshold(grayscale, 128, 192, cv2.THRESH_OTSU)
    blurred = skimage.filters.gaussian(binarized)

    # find histograms of horizontal lines, and if above certain threshold, determine upper/lower bounds of characters

    # find histograms of vertical lines, if above threshold you find regions separating individual characters. 

    # separate regions into individual sections and return. 

