
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import Adam


import matplotlib.pyplot as plt


"""
3 побудувати модель регресії для датасету із ДЗ1 (ціна будинків).
як лос використати loss = 10 * mse(x, y).
Модель має включати лише 1 hidden layer.
додати l2 регуляризацію до шару.
зберегти модель
"""
#Loss function
def loss_func(y_true=None, y_pred=None):
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return 10 * mse_loss

# Data from previous Homework
ds = pd.read_csv("kaggle/input/kc-house-data/kc_house_data.csv") #Folder Should be created!!!

print(ds.shape, ds.describe(), ds.head(), ds.info(), ds.isna().sum(), sep="\n")

ds = ds.fillna(0)
ds.isna().sum()
ds.drop("date", axis=1, inplace=True)
ds.head()
columns = abs(ds.corrwith(ds.price)) > 0.2  # those columns will have impact
X = ds.loc[:, columns].drop("price",axis=1)
y = ds.loc[:,"price"]/1000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape, X_test_scaled.shape,y_train.shape,y_test.shape,y_train[:5])

# Building model
model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=X_train.shape[1:],kernel_regularizer=l2(0.05)),
    keras.layers.Dense(1, activation="linear")
])

# Compile the model with your custom loss function
model.compile(optimizer=Adam(), loss=loss_func)
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

pd.DataFrame(history.history)[['loss', 'val_loss']].plot(figsize=(8,5))
plt.grid(True)
plt.show()

# Evaluate the model on the test set
loss = model.evaluate(X_test_scaled, y_test)
print(f'Loss on Test Set: {loss}')

# Save the model
model.save('regression_model_housing_NN.keras')


