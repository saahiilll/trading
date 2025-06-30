import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta

def preprocess_data(data, window_size=60):
    # Add technical indicators using the 'ta' library
    data['EMA'] = ta.trend.ema_indicator(close=data['Close'], window=14)
    data['RSI'] = ta.momentum.rsi(close=data['Close'], window=14)
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd_diff()  # ✅ Correct


    # Drop rows with NaNs created by indicators
    data = data.dropna()

    # Scale all features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i][0])  # Predicting the "Close" price

    return np.array(X), np.array(y), scaler
