"""
QuantConnect RESEARCH NOTEBOOK — Download Binance 5-Min OHLCV
==============================================================

Paste into a QuantConnect Research Notebook cell and run.
Generates clickable download links for each CSV directly in
the notebook output — no filesystem or Object Store needed.

INSTRUCTIONS
------------
1.  Log in to  quantconnect.com
2.  Open your project  →  click "Research"
3.  Paste this entire block into a cell  →  Run
4.  Click each download link in the output to save the CSV
5.  Place CSVs locally in:  FinRL_Crypto/data/qc_export/trade/
6.  Run locally:
        python qc_process_downloaded.py

OPTIONAL — training + validation data
--------------------------------------
Set  DOWNLOAD_TRAINVAL = True  below.  Extra download links appear
for trainval CSVs.  Save them into:
        FinRL_Crypto/data/qc_export/trainval/
and run:
        python qc_process_downloaded.py --period trainval
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
from datetime import datetime, timedelta
import base64, json
import pandas as pd
from IPython.display import display, HTML

TICKERS = [
    "AAVEUSDT", "AVAXUSDT", "BTCUSDT", "NEARUSDT", "LINKUSDT",
    "ETHUSDT",  "LTCUSDT",  "MATICUSDT", "UNIUSDT", "SOLUSDT",
]

TRADE_START = datetime(2022, 4, 30)
TRADE_END   = datetime(2022, 6, 27)

# Expected bar counts (paper: T'=16704=58×288, T=25055=87×288−1)
TRADE_BARS    = 16_704   # 58 days × 288 bars/day
TRAINVAL_BARS = 25_056   # 87 days × 288 (download full range, trim later)

DOWNLOAD_TRAINVAL = False   # set True to also grab the 25k-candle train+val window

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTBOOK SETUP
# ═══════════════════════════════════════════════════════════════════════════════
print("Initializing QuantBook …")
qb = QuantBook()
print("  QuantBook OK")

# Register each crypto symbol on Binance
symbols = {}
for tic in TICKERS:
    try:
        crypto = qb.add_crypto(tic, Resolution.MINUTE, Market.BINANCE)
        symbols[tic] = crypto.symbol
        print(f"  ✓ {tic}")
    except Exception as e:
        print(f"  ✗ {tic}: {e}")

print(f"\n  {len(symbols)}/{len(TICKERS)} symbols registered\n")


def make_download_link(csv_string, filename):
    """Create an HTML <a> tag that downloads csv_string as filename."""
    b64 = base64.b64encode(csv_string.encode()).decode()
    return (f'<a download="{filename}" '
            f'href="data:text/csv;base64,{b64}" '
            f'style="font-size:14px; margin:4px 12px; '
            f'padding:4px 10px; background:#2196F3; color:white; '
            f'text-decoration:none; border-radius:4px;">'
            f'⬇ {filename}</a>')


def fetch_and_display(label, start_dt, end_dt, expected_bars=None):
    """Fetch 1-min bars, aggregate to 5-min, display download links."""
    links = []
    summary = {}

    for tic, sym in symbols.items():
        print(f"  Fetching {tic} …", end="", flush=True)

        try:
            hist = qb.history(sym, start_dt, end_dt, Resolution.MINUTE)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if hist.empty:
            print(f"  WARNING: empty — no data returned")
            continue

        # Unpack MultiIndex if present
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.loc[sym]

        print(f"  {len(hist):,} 1-min bars …", end="", flush=True)

        # Aggregate 1-min → 5-min OHLCV
        df5 = hist.resample("5min").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

        # Trim to exact bar count — QC may include extra bars due to
        # timezone handling (ET vs UTC) or inclusive end-date behavior.
        raw_len = len(df5)
        if expected_bars and raw_len > expected_bars:
            df5 = df5.iloc[:expected_bars]
            print(f"  (trimmed {raw_len}→{expected_bars})", end="", flush=True)
        elif expected_bars and raw_len < expected_bars:
            print(f"  WARNING: only {raw_len} bars, expected {expected_bars}", end="", flush=True)

        csv_str = df5.to_csv()
        fname = f"{tic}.csv"
        links.append(make_download_link(csv_str, fname))

        summary[tic] = {
            "bars": len(df5),
            "first": str(df5.index[0]),
            "last": str(df5.index[-1]),
            "open_0": float(df5.iloc[0]["open"]),
            "close_N": float(df5.iloc[-1]["close"]),
        }
        print(f"  {len(df5):,} 5-min bars ✓")

    # Summary
    print(f"\n── {label.upper()} summary ──")
    for tic, info in summary.items():
        print(f"  {tic:12s}  {info['bars']:>6,} bars  "
              f"[{info['first'][:10]} → {info['last'][:10]}]  "
              f"open={info['open_0']:.4f}  close={info['close_N']:.4f}")

    # Manifest download link
    manifest_csv = json.dumps(summary, indent=2)
    links.append(make_download_link(manifest_csv, "manifest.json"))

    # Render clickable download links
    html = (f"<div style='padding:10px; background:#f5f5f5; "
            f"border-radius:8px; margin:10px 0;'>"
            f"<b>📁 {label.upper()} — click to download each file:</b>"
            f"<br><br>{''.join(links)}</div>")
    display(HTML(html))

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("═══════════════════════════════════════")
print("  Downloading TRADE period data")
print(f"  {TRADE_START} → {TRADE_END}")
print("═══════════════════════════════════════")
trade_summary = fetch_and_display("trade", TRADE_START, TRADE_END, TRADE_BARS)

if DOWNLOAD_TRAINVAL:
    tv_start = datetime(2022, 2, 2)   # 87 full days before trade start
    tv_end   = TRADE_START            # 2022-04-30 00:00
    print("\n═══════════════════════════════════════")
    print("  Downloading TRAINVAL period data")
    print(f"  {tv_start} → {tv_end}")
    print("═══════════════════════════════════════")
    tv_summary = fetch_and_display("trainval", tv_start, tv_end, TRAINVAL_BARS)

print("\n✓ Done!  Click the blue buttons above to download each CSV.")
