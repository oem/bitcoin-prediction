from src.data.load import load_csv
from src.data.sets import training_set, calc_test_size, test_set, create_train_test
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
df = df.iloc[::-1]
df = df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1)
sc = MinMaxScaler(feature_range=(0, 1))
dataset = sc.fit_transform(df.values.astype('float32'))
X_train, X_test, y_train, y_test = create_train_test(dataset, TEST_SIZE)

# reshape input to match [samples, time_steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], TIME_STEPS, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], TIME_STEPS, X_test.shape[1]))

# load or train the regressor
if len(sys.argv) > 1 and sys.argv[1] == "train":
    regressor = train_regressor_mae(
        X_train, y_train, BATCH_SIZE, TIME_STEPS, EPOCHS)
    regressor.save(filepath="models/01_with_mse.h5")
else:
    regressor = load_model(filepath="models/01_with_mae.h5")

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

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(predicted_train)+1, :] = predicted_train

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(predicted_train):len(dataset)-1, :] = predicted_test

# plot baseline and predictions
plt.plot(sc.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
