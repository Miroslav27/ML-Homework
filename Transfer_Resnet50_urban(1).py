import keras
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from keras import layers, models, optimizers
model_back = ResNet50(weights='imagenet', input_shape=(224,224,3), )

# Path to photos
train_data_dir = 'datasets/rural_and_urban_photos/train'
test_data_dir = 'datasets/rural_and_urban_photos/val'

# Getting classes from folders names
input_shape = (224, 224, 3)
train_classes = [folder.name for folder in Path(train_data_dir).iterdir() if folder.is_dir()]
num_classes = len(train_classes)
class_indices = {cls: idx for idx, cls in enumerate(train_classes)}

#Loading backbone and freezing it
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

# Completing the model with RESNET Backbone
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))

# Model Compile
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Training data preprocessing AUGMENTATION+Normalization
train_datagen = ImageDataGenerator(rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255 )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=input_shape[:2], batch_size=12,
                                                    class_mode='categorical', classes=train_classes)
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=input_shape[:2], batch_size=12,
                                                  class_mode='categorical', classes=train_classes)

# model training
history = model.fit(train_generator, steps_per_epoch=train_generator.samples//train_generator.batch_size, epochs=20,
          validation_data=test_generator, validation_steps=test_generator.samples//test_generator.batch_size)

# evaluating accuracy
accuracy = model.evaluate_generator(test_generator, steps=test_generator.samples//test_generator.batch_size)[1]
print(f'Test Accuracy: {accuracy}')

# plots with loss and accuracy
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Getting predictions for validation set
validation_generator = test_datagen.flow_from_directory(test_data_dir, target_size=input_shape[:2],
                                                        shuffle=False, class_mode='categorical', classes=train_classes)
predictions = model.predict_generator(validation_generator, steps=1)
predicted_labels = np.argmax(predictions, axis=1)

class_labels = {v: k for k, v in validation_generator.class_indices.items()}
labels = [class_labels[i] for i in predicted_labels]

plt.figure(figsize=(15, 7))
for i in range(len(labels)):
    plt.subplot(4, 5, i + 1)
    img = load_img(os.path.join(test_data_dir, validation_generator.filenames[i]))
    plt.imshow(img)
    plt.title(f'Predicted: {labels[i]}')
    plt.axis('off')
plt.show()