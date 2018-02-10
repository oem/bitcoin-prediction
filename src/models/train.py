from keras.preprocessing import sequence
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model, Sequential


def train_mae_1(X_train, y_train, batch_size, time_steps, epochs):
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mae")
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=2)
    return(model)
