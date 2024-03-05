import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from copy import deepcopy
def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)


def df_to_windowed_df(dataframe, n=20, f=5):
  first_date = dataframe.index[n + f]
  last_date = dataframe.index[-f]
  print(first_date, last_date, end=",")
  first_date_str = str(df[df.index == first_date].index[0]).split(" ")[0]
  last_date_str = str(df[df.index == last_date].index[0]).split(" ")[0]

  target_date = first_date
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n + f)
    # print(len(df_subset),n+f)
    if len(df_subset) != n + f:
      print(f'Error: Window of size {n + f} is too large for date {target_date}')
      return

    values = df_subset['close'].to_numpy()
    x, y = values[:-f], values[-f:]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

    if last_time:
      break

    target_date = next_date

    if target_date == last_date:
      last_time = True

  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates

  X = np.array(X)
  Y = np.array(Y)
  # print(Y)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n - i}'] = X[:, i]

  for i in range(0, f):
    ret_df[f'Target+{i}'] = Y[:, i]

  print(dataframe.symbol.iloc[0], ret_df.shape)
  return ret_df

def windowed_df_to_date_X_y(windowed_dataframe,n=20,f=5):
  df_as_np = windowed_dataframe.to_numpy()
  print(df_as_np.shape)
  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-f]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -f:]

  return dates, X.astype(np.float32), Y.astype(np.float32)

df = pd.read_csv('prices-split-adjusted.csv')

df=df.dropna(how="any")
df = df[['date','symbol', 'close']]
df['date'] = df['date'].apply(str_to_datetime)
df.index = df.pop('date')
#plt.plot(df.index[df["symbol"]=="VAR"], df['close'][df["symbol"]=="VAR"])
#plt.show()

#windowed_df = df_to_windowed_df(df[df["symbol"]=="VAR"])
#windowed_df = pd.concat([df_to_windowed_df(df[df["symbol"]==x][:-25]) for x in ['A','AAL','AAP','AAPL','ABC','ABT','ACN','ADBE','ADI','ADM']], axis=0)
windowed_df = pd.concat([df_to_windowed_df(df[df["symbol"]==x][:-25]) for x in np.unique(df.symbol)], axis=0)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

model = Sequential([layers.Input((20, 1)),
                    layers.LSTM(64),
                    layers.Dense(16, activation='relu'),
                    layers.Dropout(0.1),
                    layers.Dense(5)])

model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['mean_absolute_error'])

forward_layer = layers.LSTM(32, return_sequences=True)
backward_layer = layers.LSTM(32, activation='relu', return_sequences=True,
                      go_backwards=True)


model_bi_stacked_concat = Sequential([
    layers.Input(shape=(20, 1)),
    layers.Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='concat'),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(32,),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(5)])

model_bi_stacked_concat.compile(loss='mse',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['mean_absolute_error'])

model_bi_stacked_sum = Sequential([
    layers.Input(shape=(20, 1)),
    layers.Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode="sum"),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(32,),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(5)])

model_bi_stacked_sum.compile(loss='mse',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['mean_absolute_error'])


model.fit(X, y, epochs=10,batch_size=128, validation_split=0.2)
model.save_weights("lstm_model.weights.h5", overwrite=True)

model_bi_stacked_concat.fit(X, y, epochs=10,batch_size=128, validation_split=0.2)
model_bi_stacked_concat.save_weights("bi_stacked_concat_model.weights.h5", overwrite=True)

model_bi_stacked_sum.fit(X, y, epochs=10,batch_size=128, validation_split=0.2)
model_bi_stacked_sum.save_weights("bi_stacked_sum_model.weights.h5", overwrite=True)

for i in np.unique(df.symbol)[:10]:
  dates_val, X_val, y_val = windowed_df_to_date_X_y(df_to_windowed_df(df[df["symbol"] == i], n=20))
  val_predictions_concat = model_bi_stacked_concat.predict(X_val)  # .flatten()
  val_predictions_sum = model_bi_stacked_sum.predict(X_val)
  val_predictions = model.predict(X_val)

  plt.plot(dates_val[-50:], np.append(y_val[-46:-1, 0], y_val[-1]))
  plt.plot(dates_val[-6:], np.append(X_val[-1:, -1][0], val_predictions_concat[-5:][0]))
  plt.plot(dates_val[-6:], np.append(X_val[-1:, -1][0], val_predictions_sum[-5:][0]))
  plt.plot(dates_val[-6:], np.append(X_val[-1:, -1][0], val_predictions[-5:][0]))

  plt.legend(['Test Observations','Test bi_stacked_concat Predictions', 'Test bi_stacked_sum Predictions',
              'Test one-lstm Predictions'
              ], )
  plt.show()