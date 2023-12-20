
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras import Sequential,layers


plt.rc('figure', autolayout=True)

"""
побудувати модель класифікації для датасету https://keras.io/api/datasets/fashion_mnist/

модель кастомна/ленет. можна вчити на невеликій кількості епох
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28 , 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[:10], y_test [:10])

# Custom function to reshape the data in 1d vector
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

checkpoint_filepath = './tmp/ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_format="keras",
    save_best_only=True)

early_stopping_callback=keras.callbacks.EarlyStopping(min_delta=0.001,patience=5)

model = keras.models.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=5, activation="relu", padding="same", strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, kernel_size=5, activation="relu",padding="same", strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, activation="relu",padding="same", strides=1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
batch_size = 128
epochs = 50

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[
    early_stopping_callback,
    tensorboard_callback,
    model_checkpoint_callback
])

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
plt.show()