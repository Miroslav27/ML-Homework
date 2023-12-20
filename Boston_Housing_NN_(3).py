
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

import matplotlib.pyplot as plt
import joblib

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

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)



# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.shape, X_test_scaled.shape,y_train.shape,y_test.shape,y_train[:5])
# Building model
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

# Compile the model with your custom loss function
model.compile(optimizer=Adam(), loss=loss_func)
model.summary()
# Train the model

history = model.fit(X_train_scaled, y_train, epochs=60, batch_size=32, validation_split=0.2, verbose=2)
pd.DataFrame(history.history)[['loss', 'val_loss']].plot(figsize=(8,5))
plt.grid(True)
plt.show()
# Evaluate the model on the test set
loss = model.evaluate(X_test_scaled, y_test)
print(f'Loss on Test Set: {loss}')

# Save the model
model.save('regression_model_housing_NN.keras')


