from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from ..data.sets import prepare_data_lstm
from ..data.load import load_csv

df = load_csv("data/all_time_daily.csv")
dataset, sc, X_train, X_test, y_train, y_test = prepare_data_lstm(df, 1, 0.2)
model = load_model(filepath="models/01_with_mae.h5")

# predict
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)
predicted_train = sc.inverse_transform(predicted_train)
predicted_test = sc.inverse_transform(predicted_test)

# shift train predictions for plotting
predicted_train_plot = np.empty_like(dataset)
predicted_train_plot[:, :] = np.nan
predicted_train_plot[1:len(predicted_train)+1, :] = predicted_train

# shift test predictions for plotting
predicted_test_plot = np.empty_like(dataset)
predicted_test_plot[:, :] = np.nan
predicted_test_plot[len(predicted_train):len(dataset)-1, :] = predicted_test

# plot baseline and predictions
plt.plot(sc.inverse_transform(dataset))
plt.plot(predicted_train_plot, label='predictions on the training set')
plt.plot(predicted_test_plot, label='predictions on the test set')
plt.legend()
plt.show()
