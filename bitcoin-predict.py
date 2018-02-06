from src.data.load import load_csv
from src.data.sets import calc_test_size, calc_training_size
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model
import h5py
import math
from sklearn.metrics import mean_squared_error

BATCH_SIZE = 64
EPOCHS = 120
TIME_STEPS = 30
TEST_SIZE = 0.2

df = load_csv("./data/all_time.csv")
training_set_length = calc_training_size(len(df), BATCH_SIZE, TEST_SIZE)
train_with_padding = training_set_length + TIME_STEPS * 2
df_train = df[0:train_with_padding]
training_set = df_train.iloc[:, 1:].values
sc = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(TIME_STEPS, training_set_length + TIME_STEPS):
    X_train.append(scaled_training_set[i - TIME_STEPS:i, 0])
    y_train.append(scaled_training_set[i:i + TIME_STEPS, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

inputs_1_mae = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, 1))
lstm_1_mae = LSTM(10, stateful=True, return_sequences=True)(inputs_1_mae)
lstm_2_mae = LSTM(10, stateful=True, return_sequences=True)(lstm_1_mae)

output_1_mae = Dense(units=1)(lstm_2_mae)

regressor_mae = Model(inputs=inputs_1_mae, outputs=output_1_mae)

regressor_mae.compile(optimizer='adam', loss='mae')
for i in range(EPOCHS):
    print("Epoch: " + str(i))
    regressor_mae.fit(X_train, y_train, shuffle=False,
                      epochs=1, batch_size=BATCH_SIZE)
    regressor_mae.reset_states()

testset_length = calc_test_size(
    len(df), BATCH_SIZE, TIME_STEPS, train_with_padding)
test_with_padding = testset_length + TIME_STEPS * 2 + train_with_padding

df_test = df[train_with_padding:test_with_padding]
test_set = df_test.iloc[:, 1:].values

scaled_test_values = sc.fit_transform(test_set)

X_test = []
for i in range(TIME_STEPS, testset_length + TIME_STEPS):
    X_test.append(scaled_test_values[i - TIME_STEPS:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_mae = regressor_mae.predict(X_test, batch_size=BATCH_SIZE)
regressor_mae.reset_states()

predicted_mae = np.reshape(
    predicted_mae, (predicted_mae.shape[0], predicted_mae.shape[1]))
predicted_mae = sc.inverse_transform(predicted_mae)

y_test = []
for j in range(0, testset_length - TIME_STEPS):
    y_test = np.append(y_test, predicted_mae[j, TIME_STEPS - 1])

y_test = np.reshape(y_test, (y_test.shape[0], 1))

rmse = math.sqrt(mean_squared_error(
    test_set[TIME_STEPS:len(y_test)], y_test[0:len(y_test) - TIME_STEPS]))
print("RMSE:", rmse)

regressor_mae.save(filepath="models/01_with_mae.h5")
