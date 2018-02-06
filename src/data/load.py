import pandas as pd


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = ['DATE', 'PRICE']
    return(df)
