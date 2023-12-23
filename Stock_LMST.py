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


def df_to_windowed_df(dataframe, n=20):

  first_date = dataframe.index[n]
  last_date  = dataframe.index[-1]
  first_date_str = str(df[df.index==first_date].index[0]).split(" ")[0]
  last_date_str = str(df[df.index==last_date].index[0]).split(" ")[0]


  target_date = first_date
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)

    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
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
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]

  ret_df['Target'] = Y
  print(dataframe.symbol[0],end=",")
  return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

df = pd.read_csv('/prices-split-adjusted.csv')

df=df.dropna(how="any")
df = df[['date','symbol', 'close']]
df['date'] = df['date'].apply(str_to_datetime)
df.index = df.pop('date')
plt.plot(df.index[df["symbol"]=="VAR"], df['close'][df["symbol"]=="VAR"])

#windowed_df = df_to_windowed_df(df[df["symbol"]=="VAR"],n=10)
#windowed_df = pd.concat([df_to_windowed_df(df[df["symbol"]==x][:-7],n=10) for x in ['A','AAL','AAP','AAPL','ABC','ABT','ACN','ADBE','ADI','ADM']], axis=0)
windowed_df = pd.concat([df_to_windowed_df(df[df["symbol"]==x][:-7],n=20) for x in np.unique(df.symbol)], axis=0)

dates, X, y = windowed_df_to_date_X_y(windowed_df)


model = Sequential([layers.Input((10, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(16, activation='relu'),
                    layers.Dropout(0.1),
                    layers.Dense(1)])

model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['mean_absolute_error'])

model.fit(X, y, epochs=15, validation_split=0.15)



from copy import deepcopy

for i in np.unique(df.symbol)[:100:4]:
  dates_val, X_val, y_val = windowed_df_to_date_X_y(df_to_windowed_df(df[df["symbol"]==i],n=10))
  #model.fit(X_val[:-20],y_val[:-20],epochs=2,validation_data=[X_val[-20:],y_val[-20:]])
  val_predictions = model.predict(X_val[-20:]).flatten()

  recursive_predictions = []
  recursive_dates = dates_val[-7:]
  last_window = deepcopy(X_val[-7])
  #print(last_window)
  for target_date in recursive_dates:

    #print(last_window)
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    last_window[0:9],last_window[9] =last_window[1:10],next_prediction

  plt.plot(dates_val[-20:], val_predictions)
  plt.plot(dates_val, y_val)
  plt.plot(recursive_dates, recursive_predictions)
  plt.title(str(i))
  plt.legend(['Test Predictions', 'Test Observations','Recursive Predictions'],)
  plt.show()