from tensorflow.keras.models import load_model
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

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

for i in range(433):
    path = "preprocessed_data/dataset1/resized_images/Cars" + str(i) + ".png"
    [x1,x2,x3,x4,x5] = predict_image(path, my_model)
    image = imread(image_path)
    new_image = image[x2:x4, x3:x5, :]
    plt.imsave("licenses/license" + str(i) + ".png", new_image)
    
        


    
