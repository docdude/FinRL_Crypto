"""
Download Object Store files from QuantConnect via REST API
===========================================================

After running qc_download_binance.py as a backtest on QuantConnect,
run this script locally to pull the exported CSVs.

Setup
-----
1.  Go to  https://www.quantconnect.com/settings#/account
2.  Copy your  User ID  (numeric) and  API Token
3.  Either:
      a) Set env vars:  export QC_USER_ID=12345  QC_API_TOKEN=abc...
      b) Or enter them when prompted

Usage
-----
    python qc_download_objectstore.py
    python qc_download_objectstore.py --label trainval
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path


QC_API_BASE = "https://www.quantconnect.com/api/v2"


def get_credentials():
    """Get QC API credentials from env vars or prompt."""
    user_id = os.environ.get("QC_USER_ID", "").strip()
    api_token = os.environ.get("QC_API_TOKEN", "").strip()

    if not user_id:
        user_id = input("QuantConnect User ID: ").strip()
    if not api_token:
        api_token = input("QuantConnect API Token: ").strip()

    if not user_id or not api_token:
        sys.exit("ERROR: credentials required. Get them from "
                 "https://www.quantconnect.com/settings#/account")

    return user_id, api_token


def api_get(endpoint, auth, params=None):
    """Make authenticated GET request to QC API."""
    url = f"{QC_API_BASE}/{endpoint}"
    r = requests.get(url, auth=auth, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("success", True):
        msg = data.get("errors", data.get("messages", ["Unknown error"]))
        sys.exit(f"API error: {msg}")
    return data


def list_object_store_keys(auth):
    """List all keys in the Object Store."""
    data = api_get("object/list", auth)
    # Response: {"objects": [{"key": "...", "modified": "...", "size": ...}, ...]}
    return data.get("objects-store", data.get("objects", []))


def get_object(auth, key):
    """Download a single object by key."""
    url = f"{QC_API_BASE}/object/get"
    r = requests.get(url, auth=auth, params={"key": key}, timeout=60)
    r.raise_for_status()
    # The API returns JSON with the content
    data = r.json()
    return data.get("data", "")


def main():
    ap = argparse.ArgumentParser(
        description="Download QuantConnect Object Store CSVs")
    ap.add_argument(
        "--label", default="trade",
        choices=["trade", "trainval"],
        help="Which dataset to download (default: trade)")
    ap.add_argument(
        "--out", default="./data/qc_export",
        help="Local output directory (default: ./data/qc_export)")
    ap.add_argument(
        "--list-all", action="store_true",
        help="Just list all Object Store keys and exit")
    args = ap.parse_args()

    user_id, api_token = get_credentials()
    auth = (user_id, api_token)

    # Test connection
    print("Connecting to QuantConnect API …")
    try:
        resp = api_get("authenticate", auth)
        print(f"  Authenticated OK")
    except Exception as e:
        sys.exit(f"Authentication failed: {e}")

    # List all keys
    print("\nListing Object Store …")
    objects = list_object_store_keys(auth)

    if not objects:
        print("  Object Store is empty — did the backtest finish?")
        print("\n  Alternative: in the QC IDE, go to your project's")
        print("  left sidebar → Storage → download files manually.")
        return

    # Show all keys if requested
    if args.list_all:
        print(f"\n  {len(objects)} key(s) found:")
        for obj in objects:
            key = obj if isinstance(obj, str) else obj.get("key", obj)
            print(f"    {key}")
        return

    # Filter for our label prefix
    prefix = f"binance5m_{args.label}/"
    our_keys = []
    for obj in objects:
        key = obj if isinstance(obj, str) else obj.get("key", obj)
        if key.startswith(prefix):
            our_keys.append(key)

    if not our_keys:
        print(f"  No keys found with prefix '{prefix}'")
        print(f"  Available keys:")
        for obj in objects:
            key = obj if isinstance(obj, str) else obj.get("key", obj)
            print(f"    {key}")
        return

    print(f"  Found {len(our_keys)} file(s) for '{args.label}'")

    # Download each file
    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(our_keys):
        filename = key.split("/")[-1]
        print(f"  Downloading {key} …", end=" ", flush=True)

        content = get_object(auth, key)
        local_path = out_dir / filename

        with open(local_path, "w") as f:
            f.write(content)
        print(f"→ {local_path}")

    print(f"\nDone! Files saved to: {out_dir}/")
    print(f"Next step:  python qc_process_downloaded.py --period {args.label}")


if __name__ == "__main__":
    main()
