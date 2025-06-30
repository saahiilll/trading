from binance.client import Client
import pandas as pd
import datetime

def download_crypto_data(symbol='BTCUSDT', interval='15m', lookback_days=10):
    client = Client()

    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=lookback_days)

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_15MINUTE,
        start_str=start_time.strftime('%d %b %Y %H:%M:%S'),
        end_str=end_time.strftime('%d %b %Y %H:%M:%S')
    )

    df = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])

    df = df[['Open Time', 'Close']]
    df.set_index('Open Time', inplace=True)
    df['Close'] = pd.to_numeric(df['Close'])
    return df
