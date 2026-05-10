import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values("date")

    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)

    df["rolling_mean_7"] = df["sales"].rolling(7).mean()

    df = df.dropna()

    return df