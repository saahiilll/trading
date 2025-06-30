import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler
