from src.data.load import load_csv
from src.data.sets import create_train_test, scale_dataset, reshape_for_lstm
from src.models.train import train_regressor_mae
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import h5py
import math
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

BATCH_SIZE = 1
EPOCHS = 15
TIME_STEPS = 1
TEST_SIZE = 0.2

# prepare the training and test set
df = load_csv("./data/all_time_daily.csv")
dataset, sc = scale_dataset(df)
X_train, X_test, y_train, y_test = create_train_test(dataset, TEST_SIZE)
X_train = reshape_for_lstm(X_train, TIME_STEPS)
X_test = reshape_for_lstm(X_test, TIME_STEPS)

# load or train the regressor
if len(sys.argv) > 1 and sys.argv[1] == "train":
    regressor = train_regressor_mae(
        X_train, y_train, BATCH_SIZE, TIME_STEPS, EPOCHS)
    regressor.save(filepath="models/01_with_mse.h5")
else:
    regressor = load_model(filepath="models/01_with_mse.h5")

# predict
predicted_train = regressor.predict(X_train)
predicted_test = regressor.predict(X_test)

one_day_predict = regressor.predict(np.asarray([[y_test[-1]]]))
print(sc.inverse_transform(y_test)[-1], sc.inverse_transform(one_day_predict))

predicted_train = sc.inverse_transform(predicted_train)
predicted_test = sc.inverse_transform(predicted_test)
y_train = sc.inverse_transform(y_train)
y_test = sc.inverse_transform(y_test)

# measure
train_score = math.sqrt(mean_squared_error(
    y_train[:, 0], predicted_train[:, 0]))
test_score = math.sqrt(mean_squared_error(
    y_test[:, 0], predicted_test[:, 0]))
print("Train Score: %.2f RMSE" % train_score)
print("Test Score: %.2f RMSE" % test_score)
