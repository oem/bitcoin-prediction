from ..data.load import load_csv
from ..data.sets import prepare_data_lstm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py
import math
from sklearn.metrics import mean_squared_error
from keras.models import load_model

BATCH_SIZE = 1
EPOCHS = 15
TIME_STEPS = 1
TEST_SIZE = 0.2

# prepare the training and test set
df = load_csv("data/all_time_daily.csv")
dataset, sc, X_train, X_test, y_train, y_test = prepare_data_lstm(df, 1, 0.2)
regressor = load_model(filepath="models/01_with_mae.h5")

# predict
predicted_train = regressor.predict(X_train)
predicted_test = regressor.predict(X_test)

one_day_predict = regressor.predict(np.asarray([[y_test[-1]]]))
print(sc.inverse_transform(y_test)
      [-1], sc.inverse_transform(one_day_predict).flatten())

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
