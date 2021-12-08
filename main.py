import numpy as np
from load_data import preprocessing
from model import create_model
from ocr import train, predict
from segmentation import segmentImage

import tensorflow as tf 
import glob
import cv2

preprocessing()

file = open("boundingbox.csv")
rows = np.loadtxt(file, delimiter=",")
# use saved weights for ocr model
license_plate_detect = tf.keras.models.load_model("my_model.h5")

images = [cv2.imread(file) for file in glob.glob("preprocessed_data/resized_images/*.png")][:10]

license_plates_bounding_points = license_plate_detect.predict(images)



for index, row in enumerate(license_plates_bounding_points):

    segmented_plates = map(lambda subIm: segmentImage(images[index][subIm[2]:subIm[3],subIm[0]:subIm[1]), license_plates_bounding_points)

    for plate_nums in segmented_plates:
        all_nums = predict(plate_nums)

