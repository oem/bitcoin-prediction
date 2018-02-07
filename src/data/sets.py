from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


def calc_training_size(length, batch_size, test_percent):
    length *= 1 - test_percent
    return(int(length - (length % batch_size)))


def calc_test_size(length, batch_size, time_steps, padded_training_size):
    length -= time_steps * 2
    length -= padded_training_size
    return(int(length - (length % batch_size)))


def create_train_test(dataset, test_size):
    np.random.seed(42)
    X, y = [], []
    for i in range(len(dataset) - 1):
        X.append(dataset[i])
        y.append(dataset[i+1])
    X, y = np.asarray(X), np.asarray(y)
    return(train_test_split(X, y, test_size=test_size, shuffle=False))


def scale_dataset(df):
    df = df.iloc[::-1]  # the order of the data was sorted Date/DESC
    df = df.drop(['Date', 'Open', 'High', 'Low',
                  'Volume', 'Market Cap'], axis=1)
    sc = MinMaxScaler(feature_range=(0, 1))
    dataset = sc.fit_transform(df.values.astype('float32'))
    return(dataset, sc)


def training_set(df, batch_size, test_size, time_steps):
    training_set_length = calc_training_size(len(df), batch_size, test_size)
    train_with_padding = training_set_length + time_steps * 2
    df_train = df[0:train_with_padding]
    training_set = df_train.iloc[:, 1:].values
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_training_set = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(time_steps, training_set_length + time_steps):
        X_train.append(scaled_training_set[i - time_steps:i, 0])
        y_train.append(scaled_training_set[i:i + time_steps, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    return(X_train, y_train)


def test_set(df, batch_size, time_steps, X_train):
    padded_train_size = len(X_train) + time_steps * 2
    testset_length = calc_test_size(
        len(df), batch_size, time_steps, padded_train_size)
    test_with_padding = testset_length + time_steps * 2 + padded_train_size

    df_test = df[padded_train_size:test_with_padding]
    test_set = df_test.iloc[:, 1:].values

    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_test_values = sc.fit_transform(test_set)

    X_test = []
    for i in range(time_steps, testset_length + time_steps):
        X_test.append(scaled_test_values[i - time_steps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # yeah, this is not gonna stay here, the return signature should be X_test, y_test
    return(X_test, test_set, sc)
