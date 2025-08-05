# utils/ohlcv_resample.py

import pandas as pd

def resample_ohlcv(df, tf_minutes, timestamp_col='timestamp'):
    """
    Resample OHLCV DataFrame to any minute timeframe.
    """
    # Make sure datetime index is set
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if timestamp_col in df.columns:
            df = df.set_index(pd.to_datetime(df[timestamp_col]))
        else:
            raise ValueError("No datetime index or timestamp column found.")
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_resampled = df.resample(f'{tf_minutes}T').agg(ohlcv_dict).dropna()
    df_resampled['timestamp'] = df_resampled.index
    return df_resampled
