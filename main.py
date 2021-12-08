import numpy as np
from model import create_model
from ocr import train, predict
from segmentation import segmentImage
import tensorflow as tf 
file = open("boundingbox.csv")
rows = np.loadtxt(file, delimiter=",")
# use saved weights for ocr model
license_plate_detect = tf.keras.models.load_model("my_model.h5")
license_plates_bounding_points = license_plate_detect.predict(images)

segmented_plates = segmentImage(plates)

for plate_nums in preprocessed_plates:
    all_nums = predict(plate_nums)

