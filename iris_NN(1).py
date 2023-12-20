import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import Sequential,layers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

plt.rc('figure', autolayout=True)

"""
побудувати модель багатокласової класифікації за допомогою keras
виконання має включати

препроцесинг датасету
розбиття трейн/тест
побудова моделі керас
callback на збереження найкращої моделі
візуалізація процесу навчання за допомогою тензорборд
EarlyStopping якщо зміна лосу на valid датасеті менша, ніж 1e-3
тестування моделі на тестовому датасеті
збереження кращої моделі
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28 , 1)

# Load the data and split it between train and test sets

iris = datasets.load_iris()
X_train, X_valid, y_train, y_valid = train_test_split(iris["data"], iris["target"], train_size=0.8)


# Scale to the [0, 1] range
X_train = X_train.astype("float32") / 255
X_valid = X_valid.astype("float32") / 255

y_train = keras.utils.to_categorical(y_train, 3)
y_valid = keras.utils.to_categorical(y_valid, 3)


# Custom function to reshape the data in 1d vector

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
input_shape=(4)

checkpoint_filepath = './tmp/ckpt/iris_checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_format="keras",
    save_best_only=True)

early_stopping_callback=keras.callbacks.EarlyStopping(min_delta=0.001,patience=10)

model_iris = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(16,activation="elu"),
        #layers.Dropout(0.25),
        layers.Dense(16,activation="elu"),
        #layers.Dropout(0.25),
        layers.Dense(8,activation="elu"),
        layers.Dropout(0.25),
        layers.Dense(3, activation="softmax"),
    ]
)

model_iris.summary()
batch_size = 32
epochs = 500

model_iris.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history=model_iris.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[
    early_stopping_callback,
    tensorboard_callback,
    model_checkpoint_callback
])

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")
score = model_iris.evaluate(X_valid, y_valid, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
plt.show()
