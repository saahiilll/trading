from data_loader import download_crypto_data
from preprocessor import preprocess_data
from model_builder import build_lstm_model
from predictor import predict_next_day
from symbols import symbols  # <-- Importing your coin list

def main():
    for symbol in symbols:
        try:
            print(f"\n🔁 Processing {symbol}...")

            # Step 1: Load historical data
            data = download_crypto_data(symbol)

            # Step 2: Preprocess for LSTM
            X, y, scaler = preprocess_data(data)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], X.shape[1], 1))  # only one feature
            
            if X.shape[0] == 0:
                print(f"⚠️ Not enough data after preprocessing {symbol}, skipping.")
                continue    
                
            print(f"X shape: {X.shape}, y shape: {y.shape}")    


            # Step 3: Build and train model
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            # Step 4: Predict next-day price
            predicted_price = predict_next_day(model, scaler, X[-1])
            print(f"[{symbol}] Predicted next 15 min close price: ${predicted_price:.2f}")

        except Exception as e:
            print(f"⚠️ Skipped {symbol} due to error: {e}")

if __name__ == '__main__':
    main()
