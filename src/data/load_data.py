import pandas as pd


def load_data():
    df = pd.read_csv('../data/all_time.csv')
    df.columns = ['DATE', 'PRICE']
    return(df)
