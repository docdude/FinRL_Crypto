"""
Alpaca Crypto Data Processor

Drop-in replacement for processor_Binance.py that uses Alpaca's CryptoHistoricalDataClient
to download historical crypto OHLCV data. Works in all US states (including Texas).

The processor downloads data, computes TA-Lib technical indicators, drops correlated features,
and returns the same (data, price_array, tech_array, time_array) tuple expected by the rest
of the pipeline.

Alpaca crypto data is free and does not require API keys for historical bars.
Providing keys gives higher rate limits.
"""

import datetime as dt
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config_api import ALPACA_API_KEY, ALPACA_API_SECRET


# Map Binance-style symbols to Alpaca crypto symbols
BINANCE_TO_ALPACA = {
    'AAVEUSDT': 'AAVE/USD',
    'AVAXUSDT': 'AVAX/USD',
    'BTCUSDT':  'BTC/USD',
    'NEARUSDT': 'NEAR/USD',
    'LINKUSDT': 'LINK/USD',
    'ETHUSDT':  'ETH/USD',
    'LTCUSDT':  'LTC/USD',
    'MATICUSDT': 'MATIC/USD',
    'UNIUSDT':  'UNI/USD',
    'SOLUSDT':  'SOL/USD',
}


def _parse_timeframe(timeframe_str):
    """Convert a string like '5m', '1h', '1d' to an Alpaca TimeFrame object."""
    mapping = {
        '1m':  TimeFrame(1, TimeFrameUnit.Minute),
        '5m':  TimeFrame(5, TimeFrameUnit.Minute),
        '10m': TimeFrame(10, TimeFrameUnit.Minute),
        '15m': TimeFrame(15, TimeFrameUnit.Minute),
        '30m': TimeFrame(30, TimeFrameUnit.Minute),
        '1h':  TimeFrame(1, TimeFrameUnit.Hour),
        '2h':  TimeFrame(2, TimeFrameUnit.Hour),
        '4h':  TimeFrame(4, TimeFrameUnit.Hour),
        '1d':  TimeFrame(1, TimeFrameUnit.Day),
    }
    if timeframe_str not in mapping:
        raise ValueError(f"Timeframe '{timeframe_str}' not supported. Choose from: {list(mapping.keys())}")
    return mapping[timeframe_str]


class AlpacaProcessor:
    def __init__(self):
        self.end_date = None
        self.start_date = None
        self.tech_indicator_list = None
        self.correlation_threshold = 0.9
        self.ticker_list = None

        # Alpaca crypto historical data works without keys.
        # Authenticated requests can 401 on large queries, so use unauthenticated.
        self.client = CryptoHistoricalDataClient()

    def run(self, ticker_list, start_date, end_date, time_interval, technical_indicator_list, if_vix):
        """
        Main entry point. Downloads data, adds indicators, returns arrays.

        Returns:
            (data_df, price_array, tech_array, time_array)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

        print('Downloading data from Alpaca...')
        data = self.download_data(ticker_list, start_date, end_date, time_interval)
        print('Downloading finished! Transforming data...')

        data = self.clean_data(data)
        data = self.add_technical_indicator(data, technical_indicator_list)
        data = self.drop_correlated_features(data)

        # TA-Lib indicators produce NaN at the start of each series — fill them
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
        """Download OHLCV data from Alpaca for all tickers."""
        self.ticker_list = ticker_list
        timeframe = _parse_timeframe(time_interval)

        # Convert string dates to datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

        # Map ticker symbols to Alpaca format
        alpaca_symbols = []
        symbol_map = {}  # alpaca_symbol -> original_ticker
        for tic in ticker_list:
            alpaca_sym = BINANCE_TO_ALPACA.get(tic, tic)
            alpaca_symbols.append(alpaca_sym)
            symbol_map[alpaca_sym] = tic

        final_df = pd.DataFrame()

        for alpaca_sym in alpaca_symbols:
            original_tic = symbol_map[alpaca_sym]
            print(f'  Downloading {original_tic} ({alpaca_sym})...')

            # Alpaca may limit how much data per request; paginate if needed
            bars_df = self._fetch_bars_paginated(alpaca_sym, timeframe, start_dt, end_dt)

            if bars_df.empty:
                print(f'  WARNING: No data for {alpaca_sym}')
                continue

            bars_df['tic'] = original_tic
            # Preserve timestamp as a column before concat
            bars_df = bars_df.reset_index().rename(columns={'timestamp': 'date'})
            final_df = pd.concat([final_df, bars_df], ignore_index=True)

        # Set date as index for downstream alignment/processing
        if 'date' in final_df.columns:
            final_df = final_df.set_index('date')

        return final_df

    def _fetch_bars_paginated(self, symbol, timeframe, start_dt, end_dt):
        """Fetch bars with pagination to handle large date ranges."""
        all_bars = []
        current_start = start_dt

        while current_start < end_dt:
            # Fetch in chunks (Alpaca handles pagination internally for most cases)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=current_start,
                end=end_dt,
            )

            try:
                bars = self.client.get_crypto_bars(request)
                df = bars.df

                if df.empty:
                    break

                # Reset multi-index (symbol, timestamp) to flat DataFrame
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                    if 'symbol' in df.columns:
                        df = df.drop(columns=['symbol'])
                    df = df.set_index('timestamp')
                else:
                    df = df.reset_index()
                    df = df.set_index('timestamp')

                all_bars.append(df)
                break  # Alpaca handles pagination internally

            except Exception as e:
                print(f'  Error fetching {symbol}: {e}')
                break

        if not all_bars:
            return pd.DataFrame()

        result = pd.concat(all_bars)
        result = result[~result.index.duplicated(keep='first')]
        result = result.sort_index()

        # Keep only OHLCV columns and standardize
        col_map = {
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        }
        available = {k: v for k, v in col_map.items() if k in result.columns}
        result = result[list(available.keys())].rename(columns=available)

        # Ensure numeric types
        for col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')

        return result

    def clean_data(self, df):
        """Align all tickers to a uniform time grid using forward-fill.

        Crypto trades 24/7 but Alpaca may have gaps for lower-liquidity coins.
        Missing candles mean no trades occurred, so forward-filling the last
        known price/volume is the standard approach.
        """
        unique_tickers = df.tic.unique()

        # Build a complete 5-min grid spanning the full date range
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='5min',
        )
        print(f'  Full time grid: {len(full_index)} slots '
              f'({full_index[0]} -> {full_index[-1]})')

        aligned_frames = []
        ohlcv_cols = [c for c in df.columns if c != 'tic']

        for tic in unique_tickers:
            tic_df = df[df.tic == tic][ohlcv_cols].copy()
            before = len(tic_df)

            # Reindex to the full grid and forward-fill gaps
            tic_df = tic_df.reindex(full_index)
            tic_df = tic_df.ffill()       # carry forward last known values
            tic_df = tic_df.bfill()       # back-fill the very first rows if ticker starts late
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
        """Drop features highly correlated with others (matching Binance processor logic)."""
        # The Binance processor always drops these specific columns for consistency
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
        print(f'Technical indicators ({len(self.tech_indicator_list)}): {self.tech_indicator_list}')

        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][['close']].values
                tech_array = df[df.tic == tic][self.tech_indicator_list].values
                if_first_time = False
            else:
                price_array = np.hstack([price_array, df[df.tic == tic][['close']].values])
                tech_array = np.hstack([tech_array, df[df.tic == tic][self.tech_indicator_list].values])

        # Use the first ticker's index as the time array
        time_array = df[df.tic == self.ticker_list[0]].index

        assert price_array.shape[0] == tech_array.shape[0], \
            f"Shape mismatch: price_array {price_array.shape} vs tech_array {tech_array.shape}"

        return price_array, tech_array, time_array

    def get_TALib_features_for_each_coin(self, tic_df):
        """Compute TA-Lib indicators — identical to BinanceProcessor."""
        tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
        tic_df['macd'], _, _ = MACD(tic_df['close'], fastperiod=12,
                                    slowperiod=26, signalperiod=9)
        tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['roc'] = ROC(tic_df['close'], timeperiod=10)
        tic_df['ultosc'] = ULTOSC(tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['willr'] = WILLR(tic_df['high'], tic_df['low'], tic_df['close'])
        tic_df['obv'] = OBV(tic_df['close'], tic_df['volume'])
        tic_df['ht_dcphase'] = HT_DCPHASE(tic_df['close'])
        return tic_df
