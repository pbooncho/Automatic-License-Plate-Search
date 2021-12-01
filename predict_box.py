from tensorflow.keras.models import load_model
import numpy as np
from skimage.io import imread

my_model = load_model("my_model")

def predict_image(image_path, model):
    image = imread(image_path)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
    
    w = 224
    h = 224
    
    (startX, startY, endX, endY) = preds
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    return [image_path, startX, startY, endX, endY]
    