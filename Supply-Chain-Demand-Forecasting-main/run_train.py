from src.data_preprocessing import load_and_preprocess
from src.train_model import train_model

df = load_and_preprocess("data/sales.csv")
train_model(df)

print("Model trained and saved!")