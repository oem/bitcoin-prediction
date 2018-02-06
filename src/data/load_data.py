import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df.columns = ['DATE', 'PRICE']
    return(df)
