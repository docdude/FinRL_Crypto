#!/usr/bin/env python3
"""
Verify our Binance Vision data against Yahoo Finance for the trade period.
Cross-checks BTC and ETH close prices at daily granularity.
Also inspects the data for anomalies (gaps, zeros, NaN, stale prices).
"""

import numpy as np
import pandas as pd
import os
import sys

BASE = '/mnt/ssd_backup/FinRL_Crypto'
os.chdir(BASE)

# ---- 1. Load our Binance Vision data ----
print("=" * 80)
print("PART 1: LOAD AND INSPECT OUR BINANCE VISION DATA")
print("=" * 80)

# Load the trade period data
trade_data_path = f'{BASE}/data/trade_data/5m_22-04-30_22-06-27'
price_array = np.load(f'{trade_data_path}/price_array', allow_pickle=True)
tech_array = np.load(f'{trade_data_path}/tech_array', allow_pickle=True)
time_array = np.load(f'{trade_data_path}/time_array', allow_pickle=True)

# Also load train data for comparison
train_data_path = f'{BASE}/data/5m_25000'
price_array_train = np.load(f'{train_data_path}/price_array', allow_pickle=True)
tech_array_train = np.load(f'{train_data_path}/tech_array', allow_pickle=True)
time_array_train = np.load(f'{train_data_path}/time_array', allow_pickle=True)

print(f"\nTrade data: price_array shape = {price_array.shape}")
print(f"Trade data: tech_array shape = {tech_array.shape}")
print(f"Trade data: time_array shape = {time_array.shape}")
print(f"Trade data: time range = {time_array[0]} to {time_array[-1]}")
print(f"Trade data: number of candles = {len(time_array)}")

print(f"\nTrain data: price_array shape = {price_array_train.shape}")
print(f"Train data: time range = {time_array_train[0]} to {time_array_train[-1]}")
print(f"Train data: number of candles = {len(time_array_train)}")

# Ticker order from config_main.py
tickers = ['AAVE', 'AVAX', 'BTC', 'NEAR', 'LINK', 'ETH', 'LTC', 'MATIC', 'UNI', 'SOL']

print(f"\nTrade period prices (first candle):")
for i, ticker in enumerate(tickers):
    print(f"  {ticker:6s}: ${price_array[0, i]:>12.4f}")

print(f"\nTrade period prices (last candle):")
for i, ticker in enumerate(tickers):
    print(f"  {ticker:6s}: ${price_array[-1, i]:>12.4f}")

# ---- 2. Data quality checks ----
print("\n" + "=" * 80)
print("PART 2: DATA QUALITY CHECKS")
print("=" * 80)

# Check for NaN
nan_count = np.isnan(price_array).sum()
print(f"\nNaN in price_array: {nan_count}")

# Check for zeros
zero_count = (price_array == 0).sum()
print(f"Zeros in price_array: {zero_count}")

# Check for negative prices
neg_count = (price_array < 0).sum()
print(f"Negative prices: {neg_count}")

# Check for stale prices (same price repeated)
for i, ticker in enumerate(tickers):
    prices = price_array[:, i]
    stale = np.sum(np.diff(prices) == 0)
    stale_pct = stale / len(prices) * 100
    if stale_pct > 5:
        print(f"  WARNING: {ticker} has {stale_pct:.1f}% stale (unchanged) prices")

# Check time gaps
time_series = pd.to_datetime(time_array)
time_diffs = time_series.to_series().diff().dropna()
expected_gap = pd.Timedelta(minutes=5)
gaps = time_diffs[time_diffs != expected_gap]
if len(gaps) > 0:
    print(f"\nTime gaps (non-5min): {len(gaps)} gaps found")
    big_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=10)]
    if len(big_gaps) > 0:
        print(f"  Large gaps (>10min):")
        for idx, gap in big_gaps.items():
            print(f"    {idx}: {gap}")
else:
    print(f"\nTime gaps: NONE - all candles are exactly 5min apart")

# ---- 3. Compute return statistics for trade period ----
print("\n" + "=" * 80)
print("PART 3: PER-TICKER RETURN STATS (TRADE PERIOD)")
print("=" * 80)

print(f"\n{'Ticker':6s} {'Start':>12s} {'End':>12s} {'CumRet':>10s} {'AvgRet/5m':>12s} {'Volatility':>12s}")
print("-" * 70)
for i, ticker in enumerate(tickers):
    prices = price_array[:, i]
    cum_ret = (prices[-1] / prices[0] - 1) * 100
    rets = np.diff(prices) / prices[:-1]
    avg_ret = np.mean(rets)
    vol = np.std(rets)
    print(f"{ticker:6s} ${prices[0]:>11.2f} ${prices[-1]:>11.2f} {cum_ret:>9.2f}% {avg_ret:>12.6e} {vol:>12.6e}")

# Equal-weight portfolio return
initial_prices = price_array[0, :]
equal_weight = np.array([1e6 / len(initial_prices) / initial_prices[i] for i in range(len(initial_prices))])
portfolio_values = np.array([np.sum(equal_weight * price_array[j]) for j in range(price_array.shape[0])])
eqw_cum_ret = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
print(f"\n  EQW Portfolio: ${portfolio_values[0]:.2f} -> ${portfolio_values[-1]:.2f} = {eqw_cum_ret:.2f}%")

# ---- 4. Cross-validate against Yahoo Finance ----
print("\n" + "=" * 80)
print("PART 4: CROSS-VALIDATE AGAINST YAHOO FINANCE")
print("=" * 80)

try:
    from processor_Yahoo import Yahoofinance
    
    # Yahoo tickers for crypto
    yahoo_map = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'SOL': 'SOL-USD',
        'AVAX': 'AVAX-USD',
        'LINK': 'LINK-USD',
        'LTC': 'LTC-USD',
        'MATIC': 'MATIC-USD',
        'UNI': 'UNI-USD',
        'AAVE': 'AAVE-USD',
        'NEAR': 'NEAR-USD',
    }
    
    # Use yfinance directly for daily data comparison
    import yfinance as yf
    
    trade_start = '2022-04-30'
    trade_end = '2022-06-28'  # one day after to ensure we get all data

    print(f"\nFetching Yahoo Finance daily data for {trade_start} to {trade_end}...")
    
    for ticker_short, yahoo_ticker in yahoo_map.items():
        try:
            df = yf.download(yahoo_ticker, start=trade_start, end=trade_end, interval='1d', progress=False)
            if len(df) == 0:
                print(f"  {ticker_short}: No Yahoo data available")
                continue
                
            # Get the index in our price array
            our_idx = tickers.index(ticker_short)
            
            # For each Yahoo daily close, find the matching candle in our 5-min data
            # Match on the same date (use the last 5-min candle of each day)
            our_times = pd.to_datetime(time_array)
            our_df = pd.DataFrame({'time': our_times, 'price': price_array[:, our_idx]})
            our_df['date'] = our_df['time'].dt.date
            
            # Get last candle of each day as our "daily close"
            our_daily = our_df.groupby('date')['price'].last()
            
            yahoo_daily = df['Close'].copy()
            yahoo_daily.index = yahoo_daily.index.date
            
            # Match dates
            common_dates = sorted(set(our_daily.index) & set(yahoo_daily.index))
            
            if len(common_dates) == 0:
                print(f"  {ticker_short}: No matching dates found")
                continue
            
            # Compare prices
            diffs = []
            for d in common_dates:
                our_price = our_daily[d]
                yahoo_price = float(yahoo_daily[d])
                pct_diff = abs(our_price - yahoo_price) / yahoo_price * 100
                diffs.append(pct_diff)
            
            avg_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            print(f"  {ticker_short:6s}: {len(common_dates)} days matched, avg diff={avg_diff:.3f}%, max diff={max_diff:.3f}%")
            
            if max_diff > 5:
                print(f"    WARNING: Large price discrepancy detected!")
                for d in common_dates:
                    our_price = our_daily[d]
                    yahoo_price = float(yahoo_daily[d])
                    pct_diff = abs(our_price - yahoo_price) / yahoo_price * 100
                    if pct_diff > 2:
                        print(f"      {d}: OURS=${our_price:.2f} YAHOO=${yahoo_price:.2f} diff={pct_diff:.2f}%")

        except Exception as e:
            print(f"  {ticker_short}: Error - {e}")

except ImportError as e:
    print(f"Cannot import yfinance: {e}")
    print("Trying alternative method...")

# ---- 5. Check train data overlap/continuity with trade data ----
print("\n" + "=" * 80)
print("PART 5: TRAIN vs TRADE DATA CONTINUITY")
print("=" * 80)

train_end_time = pd.to_datetime(time_array_train[-1])
trade_start_time = pd.to_datetime(time_array[0])
gap = trade_start_time - train_end_time

print(f"Train data ends:   {train_end_time}")
print(f"Trade data starts: {trade_start_time}")
print(f"Gap between train and trade: {gap}")

# Check if train end prices match trade start prices (should be close if no gap)
print(f"\nPrice continuity check (last train candle vs first trade candle):")
for i, ticker in enumerate(tickers):
    train_last = price_array_train[-1, i]
    trade_first = price_array[0, i]
    pct_diff = abs(train_last - trade_first) / train_last * 100
    flag = " <-- GAP?" if pct_diff > 2 else ""
    print(f"  {ticker:6s}: Train=${train_last:.4f} Trade=${trade_first:.4f} diff={pct_diff:.2f}%{flag}")

# ---- 6. Check what the market actually did in this period ----
print("\n" + "=" * 80)
print("PART 6: MARKET CONTEXT (Apr 30 - Jun 27, 2022)")
print("=" * 80)
print("This was the LUNA/UST collapse + crypto winter period.")
print("BTC went from ~$38k to ~$20k. Most alts lost 40-70%.")
print("A -64.7% portfolio return in this period is PLAUSIBLE for a long-only agent")
print("that cannot short, especially if it's fully invested.")

btc_ret = (price_array[-1, tickers.index('BTC')] / price_array[0, tickers.index('BTC')] - 1) * 100
eth_ret = (price_array[-1, tickers.index('ETH')] / price_array[0, tickers.index('ETH')] - 1) * 100
sol_ret = (price_array[-1, tickers.index('SOL')] / price_array[0, tickers.index('SOL')] - 1) * 100

print(f"\nActual price changes in our data:")
print(f"  BTC: {btc_ret:.1f}%")
print(f"  ETH: {eth_ret:.1f}%")
print(f"  SOL: {sol_ret:.1f}%")
print(f"  EQW: {eqw_cum_ret:.1f}%")
print(f"  DRL: -64.7% (from backtest)")

# ---- 7. Examine the tech indicators ----
print("\n" + "=" * 80)
print("PART 7: TECH INDICATOR RANGES")
print("=" * 80)

from config_main import TECHNICAL_INDICATORS_LIST
print(f"Indicators: {TECHNICAL_INDICATORS_LIST}")
print(f"Tech array shape: {tech_array.shape}")
n_tickers = len(tickers)
n_indicators = len(TECHNICAL_INDICATORS_LIST)
print(f"Expected: {len(time_array)} x {n_tickers * n_indicators} = {len(time_array)} x {n_tickers * n_indicators}")

for j, ind in enumerate(TECHNICAL_INDICATORS_LIST):
    vals = tech_array[:, j*n_tickers:(j+1)*n_tickers]
    nans = np.isnan(vals).sum()
    print(f"  {ind:12s}: min={np.nanmin(vals):>14.4f}  max={np.nanmax(vals):>14.4f}  mean={np.nanmean(vals):>14.4f}  NaN={nans}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
