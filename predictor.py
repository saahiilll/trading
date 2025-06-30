import numpy as np

def predict_next_day(model, scaler, data, window_size=60):
    last_seq = data[-window_size:]
    pred_input = last_seq.reshape(1, window_size, 1)
    pred_scaled = model.predict(pred_input)
    return scaler.inverse_transform(pred_scaled)[0][0]
