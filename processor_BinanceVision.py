"""
Binance Vision Data Processor

Downloads historical crypto OHLCV data from Binance's public data repository
(data.binance.vision) — pre-built CSV files hosted on S3.

No API keys, no authentication, no geo-restrictions.

Uses DAILY archives (not monthly) because Binance applied kline corrections
on 2022-08-08 (see github.com/binance/binance-public-data PR #174) that
fixed 73K+ daily files but left monthly archives un-patched.  The paper
authors used the REST API after the fix, so daily archives match their data.

The processor downloads data, computes TA-Lib technical indicators, drops
correlated features, and returns the same (data, price_array, tech_array,
time_array) tuple expected by the rest of the pipeline.
"""

import io
import zipfile
from datetime import datetime, timezone
from urllib.request import urlopen

import numpy as np
import pandas as pd
from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE

# Binance Vision base URL for spot klines
# NOTE: daily archives contain the 2022-08-08 kline corrections;
#       monthly archives were never patched.
_BASE_URL = 'https://data.binance.vision/data/spot/daily/klines'

# Column names for Binance kline CSVs (no header in file)
_KLINE_COLS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
]


def _days_between(start_dt, end_dt):
    """Return a list of 'YYYY-MM-DD' strings covering start_dt to end_dt (inclusive)."""
    from datetime import timedelta
    days = []
    cur = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        days.append(cur.strftime('%Y-%m-%d'))
        cur += timedelta(days=1)
    return days


def _download_daily_zip(symbol, interval, date_str):
    """Download and extract a single daily kline CSV from Binance Vision."""
    filename = f'{symbol}-{interval}-{date_str}'
    url = f'{_BASE_URL}/{symbol}/{interval}/{filename}.zip'
    try:
        resp = urlopen(url, timeout=30)
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_name = zf.namelist()[0]
        df = pd.read_csv(zf.open(csv_name), header=None, names=_KLINE_COLS)
        return df
    except Exception as e:
        # Daily files for very recent dates may not exist yet — silently skip
        return pd.DataFrame()


class BinanceVisionProcessor:
    def __init__(self):
        self.end_date = None
        self.start_date = None
        self.tech_indicator_list = None
        self.correlation_threshold = 0.9
        self.ticker_list = None

    def run(self, ticker_list, start_date, end_date, time_interval,
            technical_indicator_list, if_vix):
        """
        Main entry point. Downloads data, adds indicators, returns arrays.

        Returns:
            (data_df, price_array, tech_array, time_array)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

        print('Downloading data from Binance Vision...')
        data = self.download_data(ticker_list, start_date, end_date, time_interval)
        print(f'Download complete: {len(data)} rows')

        data = self.clean_data(data)
        data = self.add_technical_indicator(data, technical_indicator_list)
        data = self.drop_correlated_features(data)

        # Fill NaN from TA-Lib warm-up period
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data.groupby('tic')[numeric_cols].transform(
            lambda s: s.ffill().bfill()
        )

        if if_vix:
            data = self.add_vix(data)

        price_array, tech_array, time_array = self.df_to_array(data, if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return data, price_array, tech_array, time_array

    def download_data(self, ticker_list, start_date, end_date, time_interval):
        """Download OHLCV data from Binance Vision for all tickers."""
        self.ticker_list = ticker_list

        # Map time_interval to Binance interval string
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1d',
        }
        interval = interval_map.get(time_interval, time_interval)

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        start_ms = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

        # Determine which daily files to download
        days = _days_between(start_dt, end_dt)

        final_df = pd.DataFrame()
        for tic in ticker_list:
            print(f'  Downloading {tic} ({len(days)} days)...', end='', flush=True)
            frames = []
            for day_str in days:
                df = _download_daily_zip(tic, interval, day_str)
                if not df.empty:
                    frames.append(df)

            if not frames:
                print(f' WARNING: No data for {tic}')
                continue

            tic_df = pd.concat(frames, ignore_index=True)

            # Filter to the exact requested time range
            tic_df = tic_df[(tic_df['open_time'] >= start_ms) &
                            (tic_df['open_time'] <= end_ms)]

            # Convert open_time to UTC datetime index
            tic_df['timestamp'] = pd.to_datetime(tic_df['open_time'], unit='ms', utc=True)

            # Keep only OHLCV + timestamp
            tic_df = tic_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                tic_df[col] = pd.to_numeric(tic_df[col], errors='coerce')

            tic_df['tic'] = tic
            tic_df = tic_df.set_index('timestamp')
            tic_df = tic_df[~tic_df.index.duplicated(keep='first')]
            tic_df = tic_df.sort_index()

            print(f' {len(tic_df)} bars')
            final_df = pd.concat([final_df, tic_df])

        return final_df

    def clean_data(self, df):
        """Align all tickers to a uniform time grid.

        Binance data is very complete, but small gaps may exist.
        Forward-fill any missing candles.
        """
        unique_tickers = df.tic.unique()

        # Build a complete time grid
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='5min',
        )
        print(f'  Time grid: {len(full_index)} slots '
              f'({full_index[0]} -> {full_index[-1]})')

        aligned_frames = []
        ohlcv_cols = [c for c in df.columns if c != 'tic']

        for tic in unique_tickers:
            tic_df = df[df.tic == tic][ohlcv_cols].copy()
            before = len(tic_df)
            tic_df = tic_df.reindex(full_index)
            tic_df = tic_df.ffill().bfill()
            tic_df['tic'] = tic
            filled = len(tic_df) - before
            if filled > 0:
                print(f'    {tic}: {before} -> {len(tic_df)} (filled {filled} gaps)')
            aligned_frames.append(tic_df)

        result = pd.concat(aligned_frames)
        result.index.name = 'date'
        print(f'  Aligned: {len(result)} total rows '
              f'({len(result) // len(unique_tickers)} per ticker)')
        return result

    def add_technical_indicator(self, df, tech_indicator_list):
        """Add TA-Lib indicators per coin."""
        final_df = pd.DataFrame()
        for tic in df.tic.unique():
            coin_df = df[df.tic == tic].copy()
            coin_df = self.get_TALib_features_for_each_coin(coin_df)
            final_df = pd.concat([final_df, coin_df])
        return final_df

    def drop_correlated_features(self, df):
        """Drop features highly correlated with others (matching original processor)."""
        real_drop = ['high', 'low', 'open', 'macd', 'cci', 'roc', 'willr']
        cols_to_drop = [c for c in real_drop if c in df.columns]
        print('Dropping correlated features: ', cols_to_drop)
        df_uncorrelated = df.drop(cols_to_drop, axis=1)
        return df_uncorrelated

    def add_vix(self, df):
        """Add VIX data (placeholder — not used in paper's experiments)."""
        print('VIX not used for crypto. Returning original DataFrame.')
        return df

    def df_to_array(self, df, if_vix):
        """Convert DataFrame to numpy arrays for the environment."""
        self.tech_indicator_list = list(df.columns)
        self.tech_indicator_list.remove('tic')
        print(f'Technical indicators ({len(self.tech_indicator_list)}): '
              f'{self.tech_indicator_list}')

        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][['close']].values
                tech_array = df[df.tic == tic][self.tech_indicator_list].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][['close']].values])
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][self.tech_indicator_list].values])

        time_array = df[df.tic == self.ticker_list[0]].index

        assert price_array.shape[0] == tech_array.shape[0], \
            f"Shape mismatch: price {price_array.shape} vs tech {tech_array.shape}"

        return price_array, tech_array, time_array

    def get_TALib_features_for_each_coin(self, tic_df):
        """Compute TA-Lib indicators — identical to original BinanceProcessor."""
        tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
        tic_df['macd'], _, _ = MACD(tic_df['close'], fastperiod=12,
                                    slowperiod=26, signalperiod=9)
        tic_df['cci'] = CCI(tic_df['high'], tic_df['low'],
                            tic_df['close'], timeperiod=14)
        tic_df['dx'] = DX(tic_df['high'], tic_df['low'],
                          tic_df['close'], timeperiod=14)
        tic_df['roc'] = ROC(tic_df['close'], timeperiod=10)
        tic_df['ultosc'] = ULTOSC(tic_df['high'], tic_df['low'],
                                  tic_df['close'])
        tic_df['willr'] = WILLR(tic_df['high'], tic_df['low'],
                                tic_df['close'])
        tic_df['obv'] = OBV(tic_df['close'], tic_df['volume'])
        tic_df['ht_dcphase'] = HT_DCPHASE(tic_df['close'])
        return tic_df
