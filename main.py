import numpy as np
from load_data import preprocessing
from model import create_model
from ocr import train, predict
from segmentation import segmentImage

import tensorflow as tf 

preprocessing()

file = open("boundingbox.csv")
rows = np.loadtxt(file, delimiter=",")
# use saved weights for ocr model
license_plate_detect = tf.keras.models.load_model("my_model.h5")

license_plates_bounding_points = license_plate_detect.predict(images)



for row in license_plates_bounding_points:

    segmented_plates = map(lambda subIm: segmentImage(images[i][subIm[2]:subIm[3],subIm[0]:subIm[1]), license_plates_bounding_points)

    for plate_nums in segmented_plates:
        all_nums = predict(plate_nums)

