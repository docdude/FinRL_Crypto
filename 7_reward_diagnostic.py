"""
Reward Function A/B Diagnostic

Tests the impact of the paper-vs-code reward function discrepancy:
  Paper Eq. 3:  r = v_{t+1} - v_t  (absolute portfolio change)
  Code:         r = (Δ_DRL - Δ_EQW) × norm  (relative to equal-weight)

Trains PPO with identical hyperparameters under both reward formulations
across multiple seeds to isolate the reward function's effect.

Uses two hyperparameter sets:
  A) Paper's published best (Table 2): lr=7.5e-3, gamma=0.95
  B) Our V2 CPCV best (trial #20):    lr=5e-6, gamma=0.97

Usage:
    python 7_reward_diagnostic.py                    # 3 seeds, paper params
    python 7_reward_diagnostic.py --seeds 5          # 5 seeds, paper params
    python 7_reward_diagnostic.py --seeds 3 --v2     # 3 seeds, our V2 best params
    python 7_reward_diagnostic.py --seeds 3 --both   # 3 seeds, both param sets
"""

import os
import sys
import time
import copy
import pickle
import numpy as np
import argparse
from datetime import datetime

from config_main import (
    TIMEFRAME, no_candles_for_train, no_candles_for_val,
    TICKER_LIST, SEED_CFG,
    trade_start_date, trade_end_date,
)
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from function_finance_metrics import sharpe_iid, compute_data_points_per_year


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter sets
# ─────────────────────────────────────────────────────────────────────────────

# Paper Table 2 best PPO hyperparameters
PAPER_PARAMS = {
    "learning_rate": 7.5e-3,
    "batch_size": 512,
    "gamma": 0.95,
    "net_dimension": 1024,
    "target_step": 5000,
    "eval_time_gap": 60,
    "break_step": 45000,
}

# Our V2 CPCV best (trial #20)
V2_PARAMS = {
    "learning_rate": 5e-6,
    "batch_size": 512,
    "gamma": 0.97,
    "net_dimension": 1024,
    "target_step": 2500,
    "eval_time_gap": 60,
    "break_step": 45000,
}

# Environment normalization (fixed across all experiments)
ENV_PARAMS = {
    "lookback": 1,
    "norm_cash": 2 ** -12,
    "norm_stocks": 2 ** -8,
    "norm_tech": 2 ** -15,
    "norm_reward": 2 ** -10,
    "norm_action": 10000,
}


# ─────────────────────────────────────────────────────────────────────────────
# Modified environment with absolute reward (paper Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

class CryptoEnvAbsoluteReward(CryptoEnvAlpaca):
    """
    Identical to CryptoEnvAlpaca but with absolute reward per paper Eq. 3:
        r(st, at, st+1) = v_{t+1} - v_t
    Instead of relative reward:
        r = (Δ_DRL - Δ_EQW) × norm
    
    EQW tracking is kept for metric computation but NOT used in reward.
    """

    def step(self, actions):
        self.time += 1

        # Cooldown tracking
        for i in range(len(actions)):
            if self.stocks[i] > 0:
                self.stocks_cooldown[i] += 1

        price = self.price_array[self.time]

        # CVIX risk control (same as parent)
        cvix_active = False
        if self.cvix_array is not None and self.time < len(self.cvix_array):
            cvix_val = self.cvix_array[self.time]
            if not np.isnan(cvix_val) and cvix_val > self.cvix_threshold:
                cvix_active = True
                for i in range(self.crypto_num):
                    if self.stocks[i] > 0:
                        sell_num_shares = self.stocks[i]
                        self.stocks[i] = 0
                        self.stocks_cooldown[i] = 0
                        self.cash += price[i] * sell_num_shares * (1 - self.sell_cost_pct)

        if not cvix_active:
            for i in range(self.action_dim):
                norm_vector_i = self.action_norm_vector[i]
                actions[i] = actions[i] * norm_vector_i

        actions_dollars = actions * price

        if not cvix_active:
            # Sell
            for index in np.where(actions < -self.minimum_qty_alpaca)[0]:
                if self.stocks[index] > 0:
                    if price[index] > 0:
                        sell_num_shares = min(self.stocks[index], -actions[index])
                        assert sell_num_shares >= 0, "Negative sell!"
                        self.stocks_cooldown[index] = 0
                        self.stocks[index] -= sell_num_shares
                        self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

            # Forced 5% sell every half day
            for index in np.where(self.stocks_cooldown >= 48)[0]:
                sell_num_shares = self.stocks[index] * 0.05
                self.stocks_cooldown[index] = 0
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

            # Buy
            for index in np.where(actions > self.minimum_qty_alpaca)[0]:
                if price[index] > 0:
                    fee_corrected_asset = self.cash / (1 + self.buy_cost_pct)
                    max_stocks_can_buy = (fee_corrected_asset / price[index]) * self.safety_factor_stock_buy
                    buy_num_shares = min(max_stocks_can_buy, actions[index])
                    if buy_num_shares < self.minimum_qty_alpaca[index]:
                        buy_num_shares = 0
                    self.stocks[index] += buy_num_shares
                    self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        # ── State update ──
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

        # ═══════════════════════════════════════════════════════════
        # KEY CHANGE: Absolute reward per paper Eq. 3
        # r = v_{t+1} - v_t  (transaction costs already deducted in buy/sell)
        # ═══════════════════════════════════════════════════════════
        reward = (next_total_asset - self.total_asset) * self.norm_reward

        self.total_asset = next_total_asset

        # Track EQW for metrics only (NOT used in reward)
        self.total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])

        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash

        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash

        return state, reward, done, None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(split="train"):
    if split == "train":
        data_folder = f'./data/{TIMEFRAME}_{no_candles_for_train + no_candles_for_val}'
    else:
        data_folder = f'./data/trade_data/{TIMEFRAME}_22-04-30_22-06-27'

    print(f'  Loading {split} data from: {data_folder}')
    with open(f'{data_folder}/price_array', 'rb') as f:
        price_array = pickle.load(f)
    with open(f'{data_folder}/tech_array', 'rb') as f:
        tech_array = pickle.load(f)
    with open(f'{data_folder}/time_array', 'rb') as f:
        time_array = pickle.load(f)

    print(f'    price: {price_array.shape}, tech: {tech_array.shape}')
    return price_array, tech_array, time_array


def load_cvix(time_trade):
    """Download DVOL as CVIX proxy for trade period."""
    from urllib.request import urlopen, Request
    import json as _json
    from datetime import timezone
    import pandas as pd

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
        n_above = np.nansum(cvix_array > 90.1)
        print(f'  DVOL loaded: {(~np.isnan(cvix_array)).sum()}/{len(cvix_array)} valid, '
              f'bars above 90.1: {n_above} ({n_above / len(cvix_array) * 100:.1f}%)')
        return cvix_array
    except Exception as e:
        print(f'  DVOL download failed: {e} — testing WITHOUT risk control')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Train + test
# ─────────────────────────────────────────────────────────────────────────────

def train_and_test(env_class, erl_params, env_params, price_train, tech_train,
                   price_trade, tech_trade, cvix_array, seed, label):
    """Train on full training data, test on trade period."""

    cwd = f'./train_results/reward_diag/{label}_seed{seed}'
    os.makedirs(cwd, exist_ok=True)

    print(f'\n  ── Training: {label} | seed={seed} ──')
    t0 = time.time()

    agent = DRLAgent_erl(
        env=env_class,
        price_array=price_train,
        tech_array=tech_train,
        env_params=env_params,
        if_log=False,
    )
    model = agent.get_model("ppo", gpu_id=0, model_kwargs=erl_params)
    model.random_seed = seed
    agent.train_model(model=model, cwd=cwd, total_timesteps=int(erl_params["break_step"]))

    train_time = time.time() - t0

    # ── Test ──
    env_instance = env_class(
        config={
            "price_array": price_trade,
            "tech_array": tech_trade,
            "cvix_array": cvix_array,
            "if_train": False,
        },
        env_params=env_params,
        if_log=False,
    )

    account_values = DRLAgent_erl.DRL_prediction(
        model_name="ppo",
        cwd=cwd,
        net_dimension=erl_params["net_dimension"],
        environment=env_instance,
        gpu_id=0,
    )

    # ── Metrics ──
    account_values = np.array(account_values)
    drl_cum = account_values[-1] / account_values[0] - 1
    pct_rets = account_values[1:] / account_values[:-1] - 1

    factor = compute_data_points_per_year(TIMEFRAME)
    sharpe, vol = sharpe_iid(pct_rets, bench=0, factor=factor, log=False)

    peak = np.maximum.accumulate(account_values)
    max_dd = ((account_values - peak) / peak).min()

    # EQW
    from function_finance_metrics import compute_eqw
    lookback = env_params["lookback"]
    eqw_vals, eqw_rets, eqw_cum = compute_eqw(price_trade, lookback - 1, len(price_trade) - lookback)
    eqw_cum_final = eqw_cum[-1]

    return {
        "label": label,
        "seed": seed,
        "cum_return": drl_cum,
        "sharpe": sharpe,
        "vol": vol,
        "max_drawdown": max_dd,
        "eqw_cum_return": eqw_cum_final,
        "final_value": account_values[-1],
        "train_time": train_time,
        "account_values": account_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results_rel, results_abs, param_label):
    """Print side-by-side comparison of relative vs absolute reward."""

    def stats(results, key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    hdr = f'\n{"═" * 80}\n  A/B COMPARISON: {param_label}\n{"═" * 80}'
    print(hdr)
    print(f'  {"":30} {"RELATIVE (code)":>20}  {"ABSOLUTE (paper)":>20}')
    print(f'  {"":30} {"─" * 20}  {"─" * 20}')

    for key, name, fmt in [
        ("cum_return", "Cumulative Return", "{:+.2f}%"),
        ("sharpe", "Annualized Sharpe", "{:+.4f}"),
        ("max_drawdown", "Max Drawdown", "{:+.2f}%"),
    ]:
        mult = 100 if "%" in fmt else 1
        m_r, s_r, _, _ = stats(results_rel, key)
        m_a, s_a, _, _ = stats(results_abs, key)
        val_r = fmt.format(m_r * mult) + f' ± {s_r * mult:.2f}'
        val_a = fmt.format(m_a * mult) + f' ± {s_a * mult:.2f}'
        print(f'  {name:<30} {val_r:>20}  {val_a:>20}')

    # Per-seed detail
    print(f'\n  {"Seed":<8} {"Rel CumRet":>12} {"Abs CumRet":>12} {"Δ":>10} {"Rel Sharpe":>12} {"Abs Sharpe":>12}')
    print(f'  {"─" * 8} {"─" * 12} {"─" * 12} {"─" * 10} {"─" * 12} {"─" * 12}')
    for r_rel, r_abs in zip(results_rel, results_abs):
        delta = (r_abs["cum_return"] - r_rel["cum_return"]) * 100
        print(f'  {r_rel["seed"]:<8} '
              f'{r_rel["cum_return"] * 100:>+11.2f}% '
              f'{r_abs["cum_return"] * 100:>+11.2f}% '
              f'{delta:>+9.2f}pp '
              f'{r_rel["sharpe"]:>+11.4f} '
              f'{r_abs["sharpe"]:>+11.4f}')

    # Statistical test if enough seeds
    if len(results_rel) >= 3:
        from scipy.stats import mannwhitneyu
        rels = [r["cum_return"] for r in results_rel]
        abss = [r["cum_return"] for r in results_abs]
        try:
            stat, pval = mannwhitneyu(rels, abss, alternative='two-sided')
            print(f'\n  Mann-Whitney U test (CumRet): U={stat:.1f}, p={pval:.4f}')
            if pval < 0.05:
                better = "ABSOLUTE" if np.mean(abss) > np.mean(rels) else "RELATIVE"
                print(f'  → Statistically significant difference (p<0.05): {better} is better')
            else:
                print(f'  → No statistically significant difference (p≥0.05)')
        except Exception:
            pass

    print(f'{"═" * 80}\n')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Reward function A/B diagnostic')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds (default: 3)')
    parser.add_argument('--v2', action='store_true', help='Use V2 best params instead of paper params')
    parser.add_argument('--both', action='store_true', help='Test both paper and V2 param sets')
    args = parser.parse_args()

    print(f'\n{"═" * 80}')
    print(f'  REWARD FUNCTION A/B DIAGNOSTIC')
    print(f'  Paper Eq. 3: r = Δv (absolute)  vs  Code: r = Δv_DRL - Δv_EQW (relative)')
    print(f'  Seeds: {args.seeds} starting from {SEED_CFG}')
    print(f'{"═" * 80}\n')

    # ── Load data once ──
    price_train, tech_train, time_train = load_data("train")
    price_trade, tech_trade, time_trade = load_data("trade")
    cvix_array = load_cvix(time_trade)

    # ── Determine which param sets to test ──
    param_sets = []
    if args.both:
        param_sets = [("paper_params", PAPER_PARAMS), ("v2_params", V2_PARAMS)]
    elif args.v2:
        param_sets = [("v2_params", V2_PARAMS)]
    else:
        param_sets = [("paper_params", PAPER_PARAMS)]

    all_comparisons = {}

    for param_label, erl_params in param_sets:
        print(f'\n{"━" * 80}')
        print(f'  Testing with: {param_label}')
        print(f'    lr={erl_params["learning_rate"]}, batch={erl_params["batch_size"]}, '
              f'gamma={erl_params["gamma"]}, net_dim={erl_params["net_dimension"]}, '
              f'target_step={erl_params["target_step"]}, break_step={erl_params["break_step"]}')
        print(f'{"━" * 80}')

        results_relative = []
        results_absolute = []

        for i in range(args.seeds):
            seed = SEED_CFG + i

            # A) RELATIVE reward (current code behavior)
            r_rel = train_and_test(
                env_class=CryptoEnvAlpaca,
                erl_params=erl_params,
                env_params=ENV_PARAMS,
                price_train=price_train, tech_train=tech_train,
                price_trade=price_trade, tech_trade=tech_trade,
                cvix_array=cvix_array,
                seed=seed,
                label=f"{param_label}_relative",
            )
            results_relative.append(r_rel)
            print(f'    → Relative: CumRet={r_rel["cum_return"] * 100:+.2f}%, '
                  f'Sharpe={r_rel["sharpe"]:+.4f}, time={r_rel["train_time"]:.0f}s')

            # B) ABSOLUTE reward (paper Eq. 3)
            r_abs = train_and_test(
                env_class=CryptoEnvAbsoluteReward,
                erl_params=erl_params,
                env_params=ENV_PARAMS,
                price_train=price_train, tech_train=tech_train,
                price_trade=price_trade, tech_trade=tech_trade,
                cvix_array=cvix_array,
                seed=seed,
                label=f"{param_label}_absolute",
            )
            results_absolute.append(r_abs)
            print(f'    → Absolute: CumRet={r_abs["cum_return"] * 100:+.2f}%, '
                  f'Sharpe={r_abs["sharpe"]:+.4f}, time={r_abs["train_time"]:.0f}s')

        print_comparison(results_relative, results_absolute, param_label)
        all_comparisons[param_label] = {
            "relative": results_relative,
            "absolute": results_absolute,
        }

    # ── Save ──
    out_dir = './train_results/reward_diag'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'{out_dir}/reward_ab_{ts}.pkl'

    # Strip account_values arrays to save space
    save_data = {}
    for k, v in all_comparisons.items():
        save_data[k] = {
            "relative": [{kk: vv for kk, vv in r.items() if kk != "account_values"} for r in v["relative"]],
            "absolute": [{kk: vv for kk, vv in r.items() if kk != "account_values"} for r in v["absolute"]],
        }
    with open(out_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'Results saved to: {out_path}')

    # ── Final verdict ──
    print(f'\n{"═" * 80}')
    print(f'  VERDICT')
    print(f'{"═" * 80}')
    for param_label, comp in all_comparisons.items():
        rel_mean = np.mean([r["cum_return"] for r in comp["relative"]])
        abs_mean = np.mean([r["cum_return"] for r in comp["absolute"]])
        rel_std = np.std([r["cum_return"] for r in comp["relative"]])
        abs_std = np.std([r["cum_return"] for r in comp["absolute"]])
        print(f'  {param_label}:')
        print(f'    Relative reward: {rel_mean * 100:+.2f}% ± {rel_std * 100:.2f}%')
        print(f'    Absolute reward: {abs_mean * 100:+.2f}% ± {abs_std * 100:.2f}%')
        delta = (abs_mean - rel_mean) * 100
        print(f'    Delta (abs - rel): {delta:+.2f}pp')
        if abs(delta) < abs_std * 100 + rel_std * 100:
            print(f'    → Within noise range — reward formulation likely NOT a major factor')
        else:
            better = "absolute" if delta > 0 else "relative"
            print(f'    → Outside noise range — {better} reward appears meaningfully different')
    print(f'{"═" * 80}\n')


if __name__ == '__main__':
    main()
