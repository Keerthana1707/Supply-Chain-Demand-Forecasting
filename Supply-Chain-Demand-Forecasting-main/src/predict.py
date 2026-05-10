def make_prediction(model, data):
    preds = model.predict(data)
    return preds.tolist()@app.post("/predict")
def predict():
    global model

    from data_preprocessing import load_and_preprocess
    from pytorch_forecasting import TimeSeriesDataSet

    df = load_and_preprocess("../data/sales.csv")

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="sales",
        group_ids=["sku"],
        max_encoder_length=30,
        max_prediction_length=7,
        time_varying_known_reals=["time_idx", "price", "day", "month"],
        time_varying_unknown_reals=["sales", "lag_1", "lag_7", "rolling_mean_7"],
        allow_missing_timesteps=True
    )

    loader = dataset.to_dataloader(train=False, batch_size=32)

    preds = model.predict(loader)

    return {"forecast": preds.tolist()}