def add_features(df):
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["lag_1"] = df["sales"].shift(1)
    df["rolling_mean"] = df["sales"].rolling(7).mean()
    return df.dropna()