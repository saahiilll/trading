import numpy as np

def predict_next_day(model, scaler, data, window_size=60):
    last_seq = data[-window_size:]  # shape = (60, 4)
    pred_input = last_seq.reshape(1, window_size, data.shape[1])  # ✅ 4 features
    pred_scaled = model.predict(pred_input)

    # Inverse transform needs 2D input with all features, we'll reconstruct with dummy zeros
    dummy = np.zeros((1, scaler.n_features_in_))  # shape (1, 4)
    dummy[0][0] = pred_scaled[0][0]  # Set predicted 'Close' only
    inv_price = scaler.inverse_transform(dummy)[0][0]  # Only return 'Close'

    return inv_price
