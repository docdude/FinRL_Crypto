"""
Paper Reproduction Diagnostic

Trains PPO with exact published Table 2 hyperparameters on full training data,
then tests on trade period. Compares against published metrics.

Paper methodology (Section "Training a Trading Agent"):
    1. CPCV selects best hyperparameters across 50 trials
    2. Re-train agent on whole training data with best params
    3. Test on out-of-sample trade period

This script does step 2+3 with the published best params.

Paper Table 2 PPO (CPCV) best hyperparameters:
    lr=7.5e-3, batch=512, gamma=0.95, net_dim=1024,
    target_step=5e3, break_step=4.5e4

Paper reported metrics (Table 3, PPO CPCV):
    Cumulative Return: -34.96%
    Volatility: std(R_t) where R_t = (v_t - v_{t-1}) / v_{t-1}
    Prob. of Overfitting: 8.0% (from CPCV PBO analysis, not single-run)

Usage:
    python 6_diagnostic.py              # single run with SEED_CFG
    python 6_diagnostic.py --seeds 10   # 10 seeds starting from SEED_CFG
"""

import os
import time
import pickle
import numpy as np
import argparse
from datetime import datetime

from config_main import (
    TIMEFRAME, no_candles_for_train, no_candles_for_val,
    TICKER_LIST, TECHNICAL_INDICATORS_LIST, SEED_CFG,
    trade_start_date, trade_end_date,
)
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
SEED_CFG = 0

# ─────────────────────────────────────────────────────────────────────────────
# Paper Table 2 — PPO CPCV best hyperparameters (LOCKED)
# ─────────────────────────────────────────────────────────────────────────────
PAPER_ERL_PARAMS = {
    "learning_rate": 7.5e-3,
    "batch_size": 512,
    "gamma": 0.95,
    "net_dimension": 1024,      # 2^10
    "target_step": 5000,        # 5e3 (paper Table 1: [2.5e3, 3.75e3, 5e3]; repo code has 10x bug)
    "eval_time_gap": 60,
    "break_step": 45000,        # 4.5e4
}

# Environment params (single values in search space — no choice)
PAPER_ENV_PARAMS = {
    "lookback": 1,
    "norm_cash": 2 ** -12,
    "norm_stocks": 2 ** -8,
    "norm_tech": 2 ** -15,
    "norm_reward": 2 ** -10,
    "norm_action": 10000,
}

# Paper reported metrics for comparison (Table 3)
# Paper only reports: Cumulative Return, Volatility, Prob. of Overfitting
PAPER_METRICS = {
    "cum_return": -0.3496,
    "eqw_cum_return": -0.4778,   # paper's EQW baseline
}


def load_data(split="train"):
    """Load train or trade data."""
    if split == "train":
        data_folder = f'./data/{TIMEFRAME}_{no_candles_for_train + no_candles_for_val}'
    else:
        data_folder = f'./data/trade_data/{TIMEFRAME}_22-04-30_22-06-27'

    print(f'Loading {split} data from: {data_folder}')
    with open(f'{data_folder}/price_array', 'rb') as f:
        price_array = pickle.load(f)
    with open(f'{data_folder}/tech_array', 'rb') as f:
        tech_array = pickle.load(f)
    with open(f'{data_folder}/time_array', 'rb') as f:
        time_array = pickle.load(f)

    print(f'  price: {price_array.shape}, tech: {tech_array.shape}')
    return price_array, tech_array, time_array


def compute_eqw(price_ary, indx1, indx2):
    """Equal-weight benchmark — matches function_finance_metrics.py original.

    Allocates 1e6 capital equally across coins (true equal-weight, not price-weighted).
    Note: indx1/indx2 are accepted for interface compat but unused (matches original).
    Returns use reversed division (v[:-1]/v[1:]-1) to match codebase convention.
    """
    initial_prices = price_ary[0, :]
    equal_weight = np.array([1e6 / len(initial_prices) / initial_prices[i] for i in range(len(initial_prices))])
    account_value_eqw = []
    for i in range(0, price_ary.shape[0]):
        account_value_eqw.append(np.sum(equal_weight * price_ary[i]))
    eqw_cumrets = [x / account_value_eqw[0] - 1 for x in account_value_eqw]
    account_value_eqw = np.array(account_value_eqw)
    eqw_rets_tmp = account_value_eqw[:-1] / account_value_eqw[1:] - 1
    return account_value_eqw, eqw_rets_tmp, eqw_cumrets


def train_and_test(erl_params, env_params, seed=0, no_cvix=False):
    """Train on full training data → test on trade period (paper methodology)."""

    # ── Load data ──
    price_train, tech_train, time_train = load_data("train")
    price_trade, tech_trade, time_trade = load_data("trade")

    # ── Train on FULL TRAINING data ──
    cwd = f'./train_results/cwd_tests/diagnostic_seed{seed}'
    os.makedirs(cwd, exist_ok=True)

    print(f'\n{"="*70}')
    print(f'TRAINING on full trainval data: seed={seed} | break_step={erl_params["break_step"]}')
    print(f'  lr={erl_params["learning_rate"]}, batch={erl_params["batch_size"]}, '
          f'gamma={erl_params["gamma"]}, net_dim={erl_params["net_dimension"]}')
    print(f'{"="*70}\n')

    t0 = time.time()

    agent = DRLAgent_erl(
        env=CryptoEnvAlpaca,
        price_array=price_train,
        tech_array=tech_train,
        env_params=env_params,
        if_log=True,
    )
    model = agent.get_model("ppo", gpu_id=0, model_kwargs=erl_params)
    model.random_seed = seed  # propagate seed into ElegantRL's init_before_training()
    agent.train_model(model=model, cwd=cwd, total_timesteps=int(erl_params["break_step"]))

    train_time = time.time() - t0
    print(f'\nTraining took {train_time:.1f}s')

    # ── Test on TRADE period ──
    print(f'\n{"="*70}')
    print(f'TESTING on trade period')
    print(f'{"="*70}\n')

    # Download CVIX for risk control (paper: sell all when CVIX > 90.1)
    from urllib.request import urlopen, Request
    import json as _json
    from datetime import timezone
    import pandas as pd

    cvix_array = None
    if no_cvix:
        print('  CVIX disabled (--no-cvix flag)')
    else:
      try:
        ts_start = datetime.strptime(trade_start_date[:10], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        ts_end = datetime.strptime(trade_end_date[:10], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        start_ms = int(ts_start.timestamp() * 1000)
        end_ms = int(ts_end.timestamp() * 1000)

        all_entries = []
        chunk_start = start_ms
        while chunk_start < end_ms:
            url = (f'https://www.deribit.com/api/v2/public/get_volatility_index_data'
                   f'?currency=BTC&start_timestamp={chunk_start}'
                   f'&end_timestamp={end_ms}&resolution=3600')
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urlopen(req, timeout=30)
            data = _json.loads(resp.read())
            entries = data['result']['data']
            if not entries:
                break
            all_entries.extend(entries)
            chunk_start = entries[-1][0] + 1

        seen = set()
        unique = [e for e in all_entries if e[0] not in seen and not seen.add(e[0])]
        unique.sort(key=lambda x: x[0])

        dvol_df = pd.DataFrame(unique, columns=['ts', 'open', 'high', 'low', 'close'])
        dvol_df.index = pd.to_datetime(dvol_df['ts'], unit='ms', utc=True)
        dvol_df = dvol_df[['close']].resample('5Min').interpolate(method='linear')

        time_trade_idx = pd.DatetimeIndex(time_trade)
        CVIX_df = pd.merge(
            time_trade_idx.to_series(), dvol_df['close'],
            left_index=True, right_index=True, how='left'
        )
        cvix_array = CVIX_df.iloc[:, -1].values
        print(f'  DVOL loaded: {(~np.isnan(cvix_array)).sum()}/{len(cvix_array)} valid, '
              f'range {np.nanmin(cvix_array):.1f}-{np.nanmax(cvix_array):.1f}')
        n_above = np.nansum(cvix_array > 90.1)
        print(f'  Bars above CVIX threshold 90.1: {n_above} ({n_above/len(cvix_array)*100:.1f}%)')
      except Exception as e:
        print(f'  CVIX download failed: {e} — testing WITHOUT risk control')

    env_instance = CryptoEnvAlpaca(
        config={
            "price_array": price_trade,
            "tech_array": tech_trade,
            "cvix_array": cvix_array,
            "if_train": False,
        },
        env_params=env_params,
        if_log=True,
    )

    account_values = DRLAgent_erl.DRL_prediction(
        model_name="ppo",
        cwd=cwd,
        net_dimension=erl_params["net_dimension"],
        environment=env_instance,
        gpu_id=0,
    )

    # ── Compute metrics ──
    account_values = np.array(account_values)
    lookback = env_params["lookback"]
    start_idx = lookback - 1

    # Cumulative return: R = (v - v0) / v0 (paper Section "Performance Metrics")
    drl_cumrets = [x / account_values[0] - 1 for x in account_values]
    drl_cum = drl_cumrets[-1]

    # Volatility: V = std(R_t) where R_t = (v_t - v_{t-1}) / v_{t-1} (paper definition)
    pct_rets = (account_values[1:] - account_values[:-1]) / account_values[:-1]
    vol_drl = np.std(pct_rets)

    # EQW benchmark
    end_idx = len(price_trade) - lookback
    eqw_vals, eqw_rets, eqw_cum = compute_eqw(price_trade, start_idx, end_idx)
    eqw_vals_arr = np.array(eqw_vals)
    eqw_pct_rets = (eqw_vals_arr[1:] - eqw_vals_arr[:-1]) / eqw_vals_arr[:-1]
    vol_eqw = np.std(eqw_pct_rets)

    # Max drawdown (supplementary — not in paper Table 3 but useful)
    peak = np.maximum.accumulate(account_values)
    drawdown = (account_values - peak) / peak
    max_dd = drawdown.min()

    # Cash analysis (from env log)
    cash_initial = account_values[0]

    results = {
        "seed": seed,
        "erl_params": erl_params,
        "cum_return_drl": drl_cum,
        "drl_cumrets": drl_cumrets,       # full curve for plotting
        "cum_return_eqw": eqw_cum[-1],  # last element of cumulative returns list
        "vol_drl": vol_drl,              # paper metric: std of pct returns
        "vol_eqw": vol_eqw,
        "max_drawdown": max_dd,           # supplementary (not in paper Table 3)
        "final_value": account_values[-1],
        "initial_value": account_values[0],
        "train_time": train_time,
        "n_steps": len(account_values),
    }

    return results


def print_results(results):
    """Pretty print a single result vs paper targets."""
    r = results
    p = PAPER_METRICS
    eqw_gap = r["cum_return_eqw"] - p["eqw_cum_return"]
    data_adj = r["cum_return_drl"] - p["cum_return"] - eqw_gap  # data-adjusted gap
    print(f'\n{"─"*70}')
    print(f'RESULTS: seed={r["seed"]}')
    print(f'{"─"*70}')
    print(f'  {"Metric":<25} {"Ours":>12} {"Paper":>12} {"Delta":>12}')
    print(f'  {"─"*25} {"─"*12} {"─"*12} {"─"*12}')
    print(f'  {"Cumulative Return":<25} {r["cum_return_drl"]*100:>+11.2f}% {p["cum_return"]*100:>+11.2f}% {(r["cum_return_drl"]-p["cum_return"])*100:>+11.2f}%')
    print(f'  {"Volatility (std R_t)":<25} {r["vol_drl"]:>12.6f}')
    print(f'  {"Max Drawdown":<25} {r["max_drawdown"]*100:>+11.2f}%')
    print(f'  {"─"*25} {"─"*12} {"─"*12} {"─"*12}')
    print(f'  {"EQW Cum Return":<25} {r["cum_return_eqw"]*100:>+11.2f}% {p["eqw_cum_return"]*100:>+11.2f}% {eqw_gap*100:>+11.2f}%')
    print(f'  {"EQW Volatility":<25} {r["vol_eqw"]:>12.6f}')
    fv = r["final_value"]
    print(f'  {"Final Value":<25} ${fv:>11,.0f}')
    print(f'  {"Train Time":<25} {r["train_time"]:>11.1f}s')
    print(f'  {"Data-adjusted gap":<25} {data_adj*100:>+11.2f}%  (CumRet gap minus EQW gap)')
    print(f'{"─"*70}\n')


def print_summary(all_results):
    """Print summary across seeds vs paper targets."""
    if len(all_results) <= 1:
        return
    cum_rets = [r["cum_return_drl"] for r in all_results]
    vols = [r["vol_drl"] for r in all_results]
    max_dds = [r["max_drawdown"] for r in all_results]
    p = PAPER_METRICS
    eqw_gap = all_results[0]["cum_return_eqw"] - p["eqw_cum_return"]

    print(f'\n{"═"*70}')
    print(f'SUMMARY across {len(all_results)} seeds')
    print(f'{"═"*70}')
    print(f'  {"Metric":<20} {"Mean±Std":>20} {"Best":>12} {"Paper":>12}')
    print(f'  {"─"*20} {"─"*20} {"─"*12} {"─"*12}')
    print(f'  {"Cum Return":<20} {np.mean(cum_rets)*100:+.2f}% ± {np.std(cum_rets)*100:.2f}%  '
          f'{np.max(cum_rets)*100:>+11.2f}% {p["cum_return"]*100:>+11.2f}%')
    print(f'  {"Volatility":<20} {np.mean(vols):.6f} ± {np.std(vols):.6f}  '
          f'{np.min(vols):>12.6f}')
    print(f'  {"Max Drawdown":<20} {np.mean(max_dds)*100:+.2f}% ± {np.std(max_dds)*100:.2f}%  '
          f'{np.max(max_dds)*100:>+11.2f}%')
    print(f'{"═"*70}')
    closest = min(all_results, key=lambda r: abs(r["cum_return_drl"] - p["cum_return"]))
    raw_gap = (closest["cum_return_drl"] - p["cum_return"]) * 100
    adj_gap = raw_gap - eqw_gap * 100
    print(f'  Closest to paper: seed={closest["seed"]} → CumRet={closest["cum_return_drl"]*100:+.2f}%')
    print(f'  Raw gap to paper: {raw_gap:+.2f}pp')
    print(f'  EQW baseline gap:  {eqw_gap*100:+.2f}pp (ours {all_results[0]["cum_return_eqw"]*100:+.2f}% vs paper {p["eqw_cum_return"]*100:+.2f}%)')
    print(f'  Data-adjusted gap: {adj_gap:+.2f}pp')
    print(f'{"═"*70}\n')


def main():
    parser = argparse.ArgumentParser(description='Paper reproduction diagnostic')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds to try (default: 1)')
    parser.add_argument('--no-cvix', action='store_true',
                        help='Disable CVIX risk control')
    args = parser.parse_args()

    erl_params = PAPER_ERL_PARAMS
    env_params = PAPER_ENV_PARAMS

    print(f'\n{"═"*70}')
    print(f'PAPER REPRODUCTION DIAGNOSTIC')
    print(f'{"═"*70}')
    print(f'  HPO params: lr={erl_params["learning_rate"]}, batch={erl_params["batch_size"]}, '
          f'gamma={erl_params["gamma"]}, net_dim={erl_params["net_dimension"]}')
    print(f'  target_step={erl_params["target_step"]}, break_step={erl_params["break_step"]}')
    print(f'  Seeds: {args.seeds} starting from SEED_CFG={SEED_CFG}')
    print(f'{"═"*70}\n')

    all_results = []
    for i in range(args.seeds):
        seed = SEED_CFG + i
        r = train_and_test(erl_params, env_params, seed=seed, no_cvix=args.no_cvix)
        print_results(r)
        all_results.append(r)
    print_summary(all_results)

    # Save results
    out_dir = './train_results/diagnostic'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'{out_dir}/diagnostic_{ts}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Results saved to: {out_path}')


if __name__ == '__main__':
    main()
