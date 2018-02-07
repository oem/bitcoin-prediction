from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


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


def reshape_for_lstm(data, time_steps):
    # reshape input to match [samples, time_steps, features]
    return(np.reshape(data, (data.shape[0], time_steps, data.shape[1])))


def prepare_data_lstm(df, time_steps, test_ratio):
    dataset, sc = scale_dataset(df)
    X_train, X_test, y_train, y_test = create_train_test(dataset, test_ratio)
    X_train = reshape_for_lstm(X_train, time_steps)
    X_test = reshape_for_lstm(X_test, time_steps)
    return(dataset, sc, X_train, X_test, y_train, y_test)
