"""
Process QuantConnect-exported Binance CSVs into pickled arrays
===============================================================

Reads the per-ticker CSV files you downloaded from QuantConnect's ObjectStore,
applies the same TA-Lib indicator pipeline used by processor_Binance.py, and
writes the (data, price_array, tech_array, time_array) pickles expected by
the rest of the FinRL_Crypto code.

Usage
-----
    # Trade period (default)
    python qc_process_downloaded.py

    # Train+val period
    python qc_process_downloaded.py --period trainval

    # Compare QC data against existing Binance Vision data before overwriting
    python qc_process_downloaded.py --compare

    # Custom CSV directory
    python qc_process_downloaded.py --dir /path/to/csvs

Expected directory layout
-------------------------
    data/qc_export/
        trade/
            AAVEUSDT.csv
            AVAXUSDT.csv
            BTCUSDT.csv
            ...
        trainval/          (optional)
            AAVEUSDT.csv
            ...

Each CSV must have columns: open, high, low, close, volume
with a datetime index (first column).
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from talib import RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE

from config_main import (
    TICKER_LIST, TIMEFRAME,
    trade_start_date, trade_end_date,
    no_candles_for_train, no_candles_for_val,
)


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline steps — identical logic to BinanceVisionProcessor / BinanceProcessor
# ═════════════════════════════════════════════════════════════════════════════

def load_csvs(csv_dir, ticker_list):
    """Read one CSV per ticker, combine into a single DataFrame."""
    frames = []
    for tic in ticker_list:
        path = os.path.join(csv_dir, f"{tic}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {tic}")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)

        # QC may return timestamps in ET for the request window, but the
        # actual candle times are UTC.  Localize as UTC.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Ensure OHLCV are numeric
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["tic"] = tic
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        print(f"  {tic}: {len(df):,} bars  [{df.index[0]} → {df.index[-1]}]")
        frames.append(df)

    if not frames:
        sys.exit("ERROR: no CSV files found — see --dir flag")

    combined = pd.concat(frames)
    combined.index.name = "timestamp"
    return combined


def clean_data(df):
    """Align all tickers to a uniform 5-min grid (ffill any gaps)."""
    tickers = df.tic.unique()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="5min")
    ohlcv = [c for c in df.columns if c != "tic"]

    aligned = []
    for tic in tickers:
        sub = df[df.tic == tic][ohlcv].copy()
        before = len(sub)
        sub = sub.reindex(full_idx).ffill().bfill()
        sub["tic"] = tic
        gap = len(sub) - before
        if gap:
            print(f"    {tic}: filled {gap} gaps")
        aligned.append(sub)

    out = pd.concat(aligned)
    out.index.name = "date"
    n = len(tickers)
    print(f"  Aligned: {len(out):,} rows  "
          f"({len(out) // n:,} per ticker, {n} tickers)")
    return out


def add_indicators(df):
    """Compute the 9 TA-Lib indicators (identical to BinanceProcessor)."""
    parts = []
    for tic in df.tic.unique():
        c = df[df.tic == tic].copy()
        c["rsi"]        = RSI(c["close"], timeperiod=14)
        c["macd"], _, _ = MACD(c["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        c["cci"]        = CCI(c["high"], c["low"], c["close"], timeperiod=14)
        c["dx"]         = DX(c["high"], c["low"], c["close"], timeperiod=14)
        c["roc"]        = ROC(c["close"], timeperiod=10)
        c["ultosc"]     = ULTOSC(c["high"], c["low"], c["close"])
        c["willr"]      = WILLR(c["high"], c["low"], c["close"])
        c["obv"]        = OBV(c["close"], c["volume"])
        c["ht_dcphase"] = HT_DCPHASE(c["close"])
        parts.append(c)
    return pd.concat(parts)


def drop_correlated(df):
    """Drop the same features the original processor drops."""
    drops = [c for c in ("high", "low", "open", "macd", "cci", "roc", "willr")
             if c in df.columns]
    print(f"  Dropping correlated: {drops}")
    return df.drop(columns=drops)


def df_to_arrays(df, ticker_list):
    """Convert DataFrame to (price_array, tech_array, time_array)."""
    tech_cols = [c for c in df.columns if c != "tic"]
    print(f"  Tech indicators ({len(tech_cols)}): {tech_cols}")

    first = True
    for tic in ticker_list:
        sub = df[df.tic == tic]
        if first:
            price_array = sub[["close"]].values
            tech_array  = sub[tech_cols].values
            first = False
        else:
            price_array = np.hstack([price_array, sub[["close"]].values])
            tech_array  = np.hstack([tech_array,  sub[tech_cols].values])

    time_array = df[df.tic == ticker_list[0]].index

    assert price_array.shape[0] == tech_array.shape[0], \
        f"Shape mismatch: price {price_array.shape} vs tech {tech_array.shape}"

    return price_array, tech_array, time_array


# ═════════════════════════════════════════════════════════════════════════════
# Comparison helper
# ═════════════════════════════════════════════════════════════════════════════

def _compare_prices(folder, new_price):
    """Compare new price_array against existing pickled data (before overwrite)."""
    path = os.path.join(folder, "price_array")
    if not os.path.exists(path):
        print("  No existing data to compare against.")
        return

    with open(path, "rb") as f:
        old_price = pickle.load(f)

    if old_price.shape != new_price.shape:
        print(f"  Shape mismatch: old {old_price.shape} vs new {new_price.shape}")
        return

    diff = np.abs(new_price - old_price)
    changed = (diff > 0).sum()
    print(f"  Max  price delta : {diff.max():.8f}")
    print(f"  Mean price delta : {diff.mean():.8f}")
    print(f"  Cells changed    : {changed:,} / {diff.size:,}")

    # Per-ticker breakdown (columns = tickers)
    from config_main import TICKER_LIST as TL
    for i, tic in enumerate(TL):
        col_diff = diff[:, i]
        if col_diff.max() > 0:
            print(f"    {tic:12s}  max_diff={col_diff.max():.8f}  "
                  f"changed={int((col_diff > 0).sum()):,}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def process(csv_dir, period, compare):
    print(f"\n{'=' * 64}")
    print(f"  Processing QC export:  {csv_dir}")
    print(f"  Period:                {period}")
    print(f"{'=' * 64}\n")

    # 1. Load CSVs
    print("[1/5] Loading CSVs …")
    data = load_csvs(csv_dir, TICKER_LIST)

    # 2. Clean / align
    print("\n[2/5] Cleaning & aligning to 5-min grid …")
    data = clean_data(data)

    # 3. TA-Lib indicators
    print("\n[3/5] Computing TA-Lib indicators …")
    data = add_indicators(data)

    # 4. Drop correlated features
    print("\n[4/5] Dropping correlated features …")
    data = drop_correlated(data)

    # Fill NaN from TA-Lib warm-up period
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data.groupby("tic")[num_cols].transform(
        lambda s: s.ffill().bfill()
    )

    # 5. Build arrays
    print("\n[5/5] Building numpy arrays …")
    price_array, tech_array, time_array = df_to_arrays(data, TICKER_LIST)
    tech_array[np.isnan(tech_array)] = 0

    # ── Trim trainval to exact bar count for CPCV ─────────────────────────
    # Paper: T = 25055 = 87×288−1, divisible by N=5 → 5011 per subset
    if period == "trainval":
        T_PAPER = 25_055
        n_bars = price_array.shape[0]
        if n_bars > T_PAPER:
            print(f"\n  Trimming trainval: {n_bars} → {T_PAPER} bars "
                  f"(paper T=25055, 5×5011)")
            # Drop the first bar(s) to keep the end aligned with trade_start
            trim = n_bars - T_PAPER
            price_array = price_array[trim:]
            tech_array  = tech_array[trim:]
            time_array  = time_array[trim:]
            # Also trim the DataFrame
            tickers = data.tic.unique()
            drop_idx = data.groupby("tic").head(trim).index
            data = data.drop(drop_idx)
        elif n_bars < T_PAPER:
            print(f"\n  WARNING: only {n_bars} bars, paper expects T={T_PAPER}")

    print(f"\n  price_array  {price_array.shape}")
    print(f"  tech_array   {tech_array.shape}")
    print(f"  time_array   {len(time_array)}")

    # ── EQW verification ──────────────────────────────────────────────────
    D = price_array.shape[1]
    shares = (1e6 / D) / price_array[0]
    pv = (price_array * shares).sum(axis=1)
    eqw = (pv[-1] - pv[0]) / pv[0]
    print(f"\n  *** EQW Cumulative Return : {eqw * 100:.4f}% ***")
    print(f"      Paper EQW            : -47.78%")
    print(f"      Binance Vision EQW   : -52.08%")

    # ── Determine output folder ───────────────────────────────────────────
    if period == "trade":
        folder = (f"./data/trade_data/"
                  f"{TIMEFRAME}_{trade_start_date[2:10]}_{trade_end_date[2:10]}")
    else:
        folder = f"./data/{TIMEFRAME}_{no_candles_for_train + no_candles_for_val}"

    # ── Compare before overwriting ────────────────────────────────────────
    if compare:
        print("\n── Comparison with existing data ──")
        _compare_prices(folder, price_array)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(folder, exist_ok=True)
    for name, obj in [("data_from_processor", data),
                      ("price_array", price_array),
                      ("tech_array", tech_array),
                      ("time_array", time_array)]:
        fpath = os.path.join(folder, name)
        with open(fpath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved {fpath}")

    print("\nDone!")
    return price_array, tech_array, time_array


def main():
    ap = argparse.ArgumentParser(
        description="Convert QuantConnect-exported Binance CSVs to FinRL arrays")
    ap.add_argument(
        "--dir", default="./data/qc_export",
        help="Directory containing CSVs.  If it has trade/ or trainval/ "
             "sub-folders they are used; otherwise CSVs are read directly "
             "from this directory.  (default: ./data/qc_export)")
    ap.add_argument(
        "--period", default="trade",
        choices=["trade", "trainval", "both"],
        help="Which dataset to process (default: trade)")
    ap.add_argument(
        "--compare", action="store_true",
        help="Compare QC data against existing pickled arrays before overwriting")
    args = ap.parse_args()

    periods = ["trade", "trainval"] if args.period == "both" else [args.period]

    for p in periods:
        # Try sub-folder first, fall back to --dir itself
        csv_dir = os.path.join(args.dir, p)
        if not os.path.isdir(csv_dir):
            # Check if CSVs live directly in --dir
            if any(f.endswith(".csv") for f in os.listdir(args.dir)):
                csv_dir = args.dir
            else:
                print(f"\nDirectory not found: {csv_dir}")
                print(f"Put CSVs in {csv_dir}/ or pass --dir pointing "
                      f"to the folder containing the .csv files.")
                continue
        process(csv_dir, p, args.compare)


if __name__ == "__main__":
    main()
