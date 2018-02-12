import pandas as pd


def load_csv():
    df = pd.read_csv("data/all_time_close.csv")
    return(df.iloc[:, 1:])
