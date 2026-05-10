###  Supply Chain Demand Forecasting using AI (TFT)

--

###  Project Overview

This project is an **AI-powered demand forecasting system** built for supply chain and retail businesses.
It predicts future product demand using historical sales data and deep learning.

The system uses **Temporal Fusion Transformer (TFT)** and is deployed using **FastAPI** for real-time predictions.

--

###  Problem Statement

Retail businesses often face:

 * Stockouts → Loss of revenue
 * Overstocking → Waste of capital

This project helps to:

 * Predict future demand
 * Optimize inventory
 * Reduce losses
 * 
--

###  Features

* Multi-horizon forecasting (next 7 days)
* Deep learning model (Temporal Fusion Transformer)
* Feature engineering (lags, rolling mean, date features)
* Real-time prediction using FastAPI
* Scalable and modular architecture

--

### Technologies Used

* Python
* Pandas
* NumPy
* PyTorch
* PyTorch Forecasting
* FastAPI
* VS Code

###  Project Structure

```
supply-chain-forecasting/
│
├── data/
│   └── sales.csv
│
├── src/
│   ├── main.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── tft_model.pth
│
├── run_train.py
├── requirements.txt
├── Dockerfile
└── README.md
```

--

###  Steps to Run 

1. pip install -r requirements.txt
2. Train the model:
    python run_train.py
3. Run the API:
    cd src
    uvicorn main --reload
4. Open in browser:
    http://127.0.0.1:8000/docs

--

###  Sample & Expected Output

```
{
  "status": "success",
  "forecast": [127, 128, 105, 104, 104, 103, 128]
}

* Predicted demand for next 7 days
* API-based response

```
 Output represents **predicted sales (demand)**, not price.

--

###  How It Works

1. Historical sales data is collected
2. Features like lag values and rolling averages are created
3. TFT model learns patterns (trend, seasonality, price impact)
4. Model predicts future demand
5. FastAPI serves predictions in real-time

--

### Advantages

* Accurate demand prediction
* Reduces inventory loss
* Improves supply chain efficiency
* Scalable and real-time system

--

### Limitations

* Depends on data quality
* Requires proper feature engineering
* Needs retraining with new data

--

###  Future Improvements

* Input-based prediction (SKU-wise)
* Dashboard using Streamlit
* Model accuracy metrics (WMAPE, RMSE)
* AWS deployment
* Real-time data integration

--

###  Conclusion

This project demonstrates an **end-to-end AI system** that integrates:

* Machine Learning
* Time-Series Forecasting
* API Deployment
