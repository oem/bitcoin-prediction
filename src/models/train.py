from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model


def train_regressor_mae(X_train, y_train, batch_size, time_steps, epochs):
    inputs_1_mae = Input(batch_shape=(batch_size, time_steps, 1))
    lstm_1_mae = LSTM(10, stateful=True, return_sequences=True)(inputs_1_mae)
    lstm_2_mae = LSTM(10, stateful=True, return_sequences=True)(lstm_1_mae)
    output_1_mae = Dense(units=1)(lstm_2_mae)
    regressor_mae = Model(inputs=inputs_1_mae, outputs=output_1_mae)
    regressor_mae.compile(optimizer='adam', loss='mae')
    for i in range(epochs):
        print("Epoch: " + str(i))
        regressor_mae.fit(X_train, y_train, shuffle=False,
                          epochs=1, batch_size=batch_size)
        regressor_mae.reset_states()
    return(regressor_mae)
