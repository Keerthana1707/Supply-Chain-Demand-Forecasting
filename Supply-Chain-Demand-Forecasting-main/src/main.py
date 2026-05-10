from fastapi import FastAPI
import torch

from data_preprocessing import load_and_preprocess
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

app = FastAPI()

model = None
dataset = None

@app.on_event("startup")
def load_trained_model():
    global model, dataset

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

    model = TemporalFusionTransformer.from_dataset(dataset)
    model.load_state_dict(torch.load("../models/tft_model.pth"))
    model.eval()

    print("✅ Model loaded successfully!")

@app.get("/")
def home():
    return {"message": "Model Loaded Successfully"}

@app.post("/predict")
def predict():
    global model, dataset

    try:
        loader = dataset.to_dataloader(train=False, batch_size=32)
        preds = model.predict(loader)

        preds_list = preds.tolist()

        forecast = [
            [round((x * 100) + 100, 2) for x in row]
            for row in preds_list[:10]
        ]

        return {
            "status": "success",
            "forecast": forecast
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }