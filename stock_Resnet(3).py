import keras
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

#Put your folder PATH here>>
folder_path = 'datasets/stockphoto'
images = []
# Pickking up the model RESNET50
model = ResNet50(weights='imagenet', input_shape=(224,224,3))

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img_array = preprocess_image(img_path)

        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions)[0]

        # Extract top 3 predictions
        top3_predictions = decoded_predictions[:3]

        # Display the image
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('off')

        # Print the top 3 predicted labels and probabilities on new lines
        text=("\n").join([f'{class_name}: {score:.2%}' for i, (label, class_name, score) in enumerate(top3_predictions)])
        plt.title(f'Top 3 Predictions:\n{text}')

        # Show the plot
        plt.show()
