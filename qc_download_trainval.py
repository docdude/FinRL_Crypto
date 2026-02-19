"""
QuantConnect RESEARCH NOTEBOOK — Download Binance 5-Min OHLCV (TRAINVAL)
=========================================================================

Paste into a QuantConnect Research Notebook cell and run.
Generates clickable download links for each CSV directly in
the notebook output — no filesystem or Object Store needed.

This downloads the TRAINING + VALIDATION window:
    2022-02-02 00:00  →  2022-04-30 00:00  UTC   (87 days × 288 bars = 25,056 bars)
    Processing trims to T = 25,055 = 87×288−1 (paper spec, evenly divisible by N=5)

Because this is a large window (~87 days), the data is fetched in
weekly chunks to avoid QC timeout errors.

INSTRUCTIONS
------------
1.  Log in to  quantconnect.com
2.  Open your project  →  click "Research"
3.  Paste this entire block into a cell  →  Run
4.  Click each download link in the output to save the CSV
5.  Place CSVs locally in:  FinRL_Crypto/binance_data_trainval/
6.  Run locally:
        python qc_process_downloaded.py --dir ./binance_data_trainval --period trainval
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

# Trainval window: 87 full days before trade start (02/02 → 04/30)
TRADE_START = datetime(2022, 4, 30)
TV_START    = datetime(2022, 2, 2)                              # 2022-02-02 00:00
TV_END      = TRADE_START                                       # 2022-04-30 00:00

CHUNK_DAYS      = 7       # Fetch in weekly chunks to avoid QC timeouts
EXPECTED_BARS   = 25_056  # 87 days × 288 bars/day (full range; trimmed to 25055 during processing)

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


def fetch_chunked(tic, sym, start_dt, end_dt, chunk_days):
    """Fetch 1-min bars in weekly chunks, return concatenated 5-min DataFrame."""
    all_5m = []
    cursor = start_dt
    chunk = 1

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        print(f"    chunk {chunk}: {cursor.strftime('%Y-%m-%d %H:%M')} → "
              f"{chunk_end.strftime('%Y-%m-%d %H:%M')} …", end="", flush=True)

        try:
            hist = qb.history(sym, cursor, chunk_end, Resolution.MINUTE)
        except Exception as e:
            print(f"  ERROR: {e}")
            cursor = chunk_end
            chunk += 1
            continue

        if hist.empty:
            print(f"  (empty)")
            cursor = chunk_end
            chunk += 1
            continue

        # Unpack MultiIndex if present
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.loc[sym]

        # Aggregate 1-min → 5-min
        df5 = hist.resample("5min").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

        all_5m.append(df5)
        print(f"  {len(df5):,} bars")

        cursor = chunk_end
        chunk += 1

    if not all_5m:
        return pd.DataFrame()

    combined = pd.concat(all_5m)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    # Trim to exact bar count — QC may over-fetch due to TZ / end-date quirks
    if len(combined) > EXPECTED_BARS:
        print(f"    trimming {len(combined)} → {EXPECTED_BARS} bars")
        combined = combined.iloc[:EXPECTED_BARS]
    elif len(combined) < EXPECTED_BARS:
        print(f"    WARNING: only {len(combined)} bars, expected {EXPECTED_BARS}")

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("═══════════════════════════════════════════════════════")
print("  Downloading TRAINVAL period data")
print(f"  {TV_START} → {TV_END}")
print(f"  ({CHUNK_DAYS}-day chunks to avoid timeouts)")
print("═══════════════════════════════════════════════════════\n")

links = []
summary = {}

for tic, sym in symbols.items():
    print(f"  [{tic}]")
    df5 = fetch_chunked(tic, sym, TV_START, TV_END, CHUNK_DAYS)

    if df5.empty:
        print(f"    WARNING: no data for {tic}!")
        continue

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
    print(f"    TOTAL: {len(df5):,} 5-min bars  "
          f"[{df5.index[0]} → {df5.index[-1]}]")
    print()

# Summary
print(f"\n── TRAINVAL summary ──")
for tic, info in summary.items():
    print(f"  {tic:12s}  {info['bars']:>6,} bars  "
          f"[{info['first'][:16]} → {info['last'][:16]}]  "
          f"open={info['open_0']:.4f}  close={info['close_N']:.4f}")

# Manifest download link
manifest_csv = json.dumps(summary, indent=2)
links.append(make_download_link(manifest_csv, "manifest.json"))

# Render clickable download links
html = (f"<div style='padding:10px; background:#f5f5f5; "
        f"border-radius:8px; margin:10px 0;'>"
        f"<b>📁 TRAINVAL — click to download each file:</b>"
        f"<br><br>{''.join(links)}</div>")
display(HTML(html))

print("\n✓ Done!  Click the blue buttons above to download each CSV.")
print(f"\nAfter downloading, place all CSVs in  binance_data_trainval/  and run:")
print(f"  python qc_process_downloaded.py --dir ./binance_data_trainval --period trainval")
