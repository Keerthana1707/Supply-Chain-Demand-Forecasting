from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from lightning.pytorch import Trainer
import torch

def train_model(df):

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

    train_loader = dataset.to_dataloader(train=True, batch_size=32)

    model = TemporalFusionTransformer.from_dataset(dataset)

    trainer = Trainer(max_epochs=5)
    trainer.fit(model, train_loader)

    torch.save(model.state_dict(), "models/tft_model.pth")

    return model