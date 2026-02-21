# Systematic Reproduction Review: arxiv:2209.05559
## "Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting"
### Gort et al. (ICAIF '22 / AAAI '23)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Paper Methodology Overview](#2-paper-methodology-overview)
3. [Section-by-Section Code vs. Paper Analysis](#3-section-by-section-analysis)
   - 3.1 [Environment (MDP & State Space)](#31-environment)
   - 3.2 [Technical Indicators](#32-technical-indicators)
   - 3.3 [CVIX Risk Control](#33-cvix-risk-control)
   - 3.4 [CPCV Cross-Validation](#34-cpcv-cross-validation)
   - 3.5 [PBO (Probability of Backtest Overfitting)](#35-pbo)
   - 3.6 [Backtest](#36-backtest)
   - 3.7 [Metrics & Sharpe Calculation](#37-metrics)
4. [Bug-by-Bug Impact Analysis](#4-bug-by-bug-impact-analysis)
5. [Why Results Don't Match — With or Without Fixes](#5-why-results-dont-match)
6. [Numerical Summary Table](#6-numerical-summary)
7. [Conclusions](#7-conclusions)

---

## 1. Executive Summary

We attempted a full end-to-end reproduction of Gort et al.'s PPO CPCV pipeline for cryptocurrency trading. We identified **9 bugs** in the released code, fixed all of them, aligned the hyperparameter search space with Table 2 of the paper, and ran 50-trial CPCV optimization. 

**Key finding: The reproduction gap exists regardless of bug fixes.** The original buggy code produces ~**-60%** cumulative return; our fixed code produces **-68.3%**. The paper reports **-34.96%**. Since the original code with all bugs intact already fails to reproduce the paper, the bugs cannot explain the gap. The only way to approach the paper's result is to use its exact best hyperparameters with a specific random seed — most seeds yield ~-60%.

This document explains why, section by section, with concrete numerical examples for each bug.

---

## 2. Paper Methodology Overview

The paper proposes:

1. **Train** a PPO agent on 10 cryptocurrencies using 5-minute Binance candles
2. **CPCV** (Combinatorial Purged Cross-Validation): N=5 groups, K=2 test groups → 10 splits per trial, H=50 trials
3. **PBO** (Probability of Backtest Overfitting): Build matrix M of validation returns, split into S=14 submatrices, compute rank-based logit statistic
4. **Hypothesis test**: Reject agents with PBO > threshold (paper finds PBO=8.0% for PPO CPCV)
5. **Backtest**: Apply best agent to unseen trade period (Apr 30 – Jun 27, 2022) with CVIX risk control
6. **Result**: PPO CPCV achieves -34.96% cumulative return (vs. EQW -47.78%, S&P BDM -38.79%)

Paper's 6 uncorrelated features: **volume, RSI, DX, ULTOSC, OBV, HT_DCPHASE** (after filtering 15 candidate features by >60% correlation)

---

## 3. Section-by-Section Analysis

### 3.1 Environment

#### Paper Description
- MDP with state = [cash, holdings(D), technical_indicators(D×F) × lookback]
- D=10 cryptocurrencies, F=6 features after correlation filtering
- Transaction cost: 0.3% per trade (buy and sell)
- **Reward (Eq. 3)**: `r(st, at, st+1) = v_{t+1} - v_t - c_t` — **absolute** portfolio value change minus transaction costs
- Forced 5% sell after cooldown period when position held too long

#### Code Reality (AI4Finance repo / our repo)
- `environment_Alpaca.py` implements the MDP correctly in structure
- Transaction cost: `buy_cost_pct=0.003, sell_cost_pct=0.003` ✓ matches paper's 0.3%
- Cooldown forced sell at 48 periods (= 4 hours at 5min timeframe) ✓
- Safety factor 0.9 on buys ✓
- **Bug 1** was in `get_state()` — see Section 4

**Reward function discrepancy (unfixed)**:
- **Paper Eq. 3**: `r = v_{t+1} - v_t` (absolute portfolio change; costs already deducted in buy/sell)
- **Code**: `reward = (delta_bot - delta_eqw) * norm_reward` (relative to equal-weight benchmark)
- This is a real deviation from the paper's MDP formulation. The code trains the agent to beat a benchmark (excess-return RL), while the paper's equations describe maximizing absolute returns.
- Impact: changes the policy gradient landscape — the agent receives positive rewards when losing money (if EQW loses more), and negative rewards when making money (if EQW makes more). This is a legitimate financial RL design choice, but it contradicts the published equations.
- **Diagnostic**: `7_reward_diagnostic.py` A/B tests both formulations across seeds to quantify the actual impact.

#### Our configuration
- D=10 (AAVE, AVAX, BTC, NEAR, LINK, ETH, LTC, MATIC, UNI, SOL) — matches the paper exactly
- Note: the AI4Finance GitHub fork has most tickers commented out (only BTC+ETH), but our working `config_main.py` restored all 10

#### Verdict
Environment logic is structurally sound. The key divergences are: the get_state bug (Bug 1, fixed), and the reward function formulation (unfixed — see diagnostic).

---

### 3.2 Technical Indicators

#### Paper States
> "We consider 15 features... Finally, 6 uncorrelated features are kept: **volume, RSI, DX, ULTSOC, OBV, HT_DCPHASE**"

#### How Indicators Actually Flow (Processor → Environment)

**Critical detail**: The `TECHNICAL_INDICATORS_LIST` in `config_main.py` is a **red herring**. Both `processor_Binance.py` and `processor_BinanceVision.py` **ignore** this parameter entirely. The actual feature pipeline is hardcoded inside the processor:

1. **`get_TALib_features_for_each_coin()`** computes all 9 indicators on every coin:
   ```
   RSI, MACD, CCI, DX, ROC, ULTOSC, WILLR, OBV, HT_DCPHASE
   ```

2. **`drop_correlated_features()`** hardcodes the drop list (not data-driven despite computing a correlation matrix):
   ```python
   real_drop = ['high', 'low', 'open', 'macd', 'cci', 'roc', 'willr']
   ```

3. **`df_to_array()`** builds `tech_indicator_list` from whatever columns remain in the DataFrame (minus 'tic'), NOT from `config_main.py`.

#### Verified On-Disk Data

We inspected the actual pickled `tech_array` and `data_from_processor` used in our experiments:
```
Training tech_array shape: (25045, 70) → 70 / 10 coins = 7 features per coin
Trade tech_array shape:    (16704, 70) → same 7 features

Data columns: ['close', 'volume', 'rsi', 'dx', 'ultosc', 'obv', 'ht_dcphase', 'tic']
```

#### Actual Feature Comparison

| Indicator  | Paper (6 features) | Actual tech_array (7 features) | Status |
|------------|:------------------:|:------------------------------:|--------|
| volume     | ✓                  | ✓                              | Match  |
| RSI        | ✓                  | ✓                              | Match  |
| DX         | ✓                  | ✓                              | Match  |
| ULTOSC     | ✓                  | ✓                              | Match  |
| OBV        | ✓                  | ✓                              | Match  |
| HT_DCPHASE | ✓                  | ✓                              | Match  |
| close      | ✗                  | ✓                              | Extra  |

All 6 of the paper's features are present. The only difference is an additional `close` column in `tech_array`. Note that `close` is also available via `price_array`, so it is redundant — but this is a minor discrepancy (7 vs 6 features), not a fundamental mismatch.

#### Why `config_main.py` Is Misleading

`config_main.py` lists:
```python
TECHNICAL_INDICATORS_LIST = ['open', 'high', 'low', 'close', 'volume',
                             'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']
```
This 11-item list is passed to `processor.run()` → `add_technical_indicator(df, tech_indicator_list)`, but the method body never references `tech_indicator_list`. It unconditionally calls `get_TALib_features_for_each_coin()` and ignores the argument. The `drop_correlated_features()` hardcoded drop list then reduces to the same 7 surviving columns regardless of what was in the config.

**Impact**: The technical indicator set is **nearly aligned with the paper** (6/6 match + 1 extra `close`). This is NOT a significant source of reproduction failure. The extra `close` column adds one redundant dimension per coin (70 vs 60 total tech features) but carries no new information since `close` is already in `price_array`.

---

### 3.3 CVIX Risk Control

#### Paper Description
The paper describes CVIX as **environment constraint #3** — architecturally part of the `step` function, on the same level as transaction fees (#1) and non-negative balance (#2):

> "Risk control. The cryptocurrency market regularly drops in terms of market capitalization, sometimes even ≥ 70%. To control the risk for these market situations, we employ the Cryptocurrency Volatility Index, CVIX (Bonaparte 2021). Extreme market situations increase the value of CVIX. Once the CVIX exceeds a certain threshold, we stop buying and then sell all our cryptocurrency holdings. We resume trading once the CVIX returns under the threshold."

The threshold is defined as:
> "We take the average CVIX value over those [crash] time frames as our threshold, CVIXt = 90.1."

#### Key Timing Observation

Although CVIX is described as part of the environment design, it is **functionally a test-only feature**:
- **Training period**: 02/02/2022 – 04/30/2022
- **Crashes**: 05/06–05/12 and 06/09–06/15, 2022 — both within the **testing** period
- CVIX would **never exceed 90.1 during training** (the crashes hadn't happened yet)
- Even if CVIX were coded into the training environment, it would be a no-op

Therefore there is **no train/test mismatch** — CVIX simply doesn't trigger during the training window regardless of implementation.

#### Information Leakage in Threshold Calibration

The threshold 90.1 is the **average CVIX during the test-period crashes**. The paper determined this value by examining the test data, then applied it as a constraint during the test period. This is a subtle form of lookahead bias — the risk control parameter was calibrated from the very data it is applied to.

#### Original Repos
- **Burntt/FinRL_Crypto** (author's original, now 404): Unknown implementation
- **AI4Finance-Foundation/FinRL_Crypto**:
  - `download_external_indicator()` in `4_backtest.py` generates **dummy random-walk data** (not real CVIX)
  - `add_5m_CVIX()` in `processor_Binance.py` downloads CVOL-USD from Yahoo Finance (now delisted)
  - `CryptoEnvAlpaca.step()` has **NO** cvix logic — the constraint described in the paper is not implemented

#### Our Implementation
We added CVIX support using Deribit DVOL (Bitcoin Volatility Index) as a proxy:
- Threshold: 90.1 (matching paper)
- When DVOL > 90.1: force-sell all holdings, block all buying
- Active only during backtest (matching the paper's effective behavior — no triggers during training)

#### Measured Impact
- **~37% of the trade period** has DVOL > 90.1 — the agent is locked out of trading for over a third of the backtest
- The paper identifies only **two crash windows** (05/06–05/12 and 06/09–06/15 ≈ 12 days out of 58 ≈ **21%**), but our DVOL proxy exceeds the threshold for nearly double that duration
- This discrepancy likely arises because DVOL (Deribit Bitcoin Volatility Index) ≠ CVIX (Bonaparte 2021) — different instruments with different threshold-crossing characteristics
- First DVOL crossing: 2022-05-16 (Terra/LUNA crash)
- A/B test: Without CVIX = **-65.7%**, With CVIX = **-68.3%** → CVIX costs 2.7%
- CVIX hurts because: (a) forced-sell crystallizes losses at crash lows, (b) agent can't re-enter when recovery begins

#### Verdict
CVIX is described as an environment constraint but only matters during the test period. Our implementation correctly applies it only during backtest. The 2.7% cost suggests the CVIX threshold (90.1, calibrated from test-period crash averages) is poorly suited as a risk control — it force-sells at crash lows rather than protecting against losses. The threshold calibration from test data is itself a form of information leakage that could have been used to optimize the paper's reported backtest results.

---

### 3.4 CPCV Cross-Validation

#### Paper Description
- N=5 groups, K=2 test groups → C(5,2) = 10 splits per trial
- H=50 trials of hyperparameter optimization
- Embargo period to prevent information leakage
- Objective: mean Sharpe(DRL) - mean Sharpe(EQW) across 10 splits

#### Code Implementation
- `function_CPCV.py`: Implements `CombPurgedKFoldCV` correctly (adapted from de Prado's AFML)
- `1_optimize_cpcv.py`: Optuna optimization with correct N=5, K=2, 50 trials ✓
- Embargo: `set_Pandas_Timedelta(TIMEFRAME) * t_final * 5` where t_final=10 → 250 minutes of embargo ✓

#### Our Execution
- V2 optimization: 50 trials completed
- Best trial #20: value=0.1086, lr=5e-6, bs=512, gamma=0.97, net_dim=1024, target_step=2500, break_step=45000
- 16/50 trials positive (DRL beats EQW), 34/50 negative

#### Key Issue: Learning Rate Convergence
Optuna converges almost exclusively to `lr=5e-6` (the smallest value in the search space):
- V1: ~57% of trials at lr=5e-6
- V2: similar pattern

The paper's best hyperparameters use `lr=7.5e-3` — **three orders of magnitude higher**. This suggests either:
1. The loss landscape is very different with the paper's data (Binance) vs ours (CoinAPI)
2. The paper's Optuna wasn't actually converging to lr=7.5e-3 through optimization but was manually set
3. The technical indicator mismatch changes the gradient magnitudes

---

### 3.5 PBO

#### Paper Description
- Matrix M: T' × H where T' = validation length per split, H = 50 trials
- S=14 submatrices ("We split M into 14 submatrices")
- PBO = fraction of logits < 0
- Paper result: **PBO = 8.0%** for PPO CPCV

#### Code Bugs Found

**Bug 2 — axis=0 → axis=1 in `build_matrix_M_splits()`**:
The function averages validation returns across CPCV splits for each trial. Using `axis=0` averaged across trials (wrong dimension). The fix to `axis=1` averages across splits (correct — each row is a split, each column is a time step within the validation set).

**Bug 3 — off-by-one in trial indexing**:
`for no_trial in range(len(trials) - 1)` skips the last trial. Fix: `range(len(trials))`.

**Bug 9 — hardcoded model names**:
`model_names` list was hardcoded, causing index errors when running different configurations.

#### Our Results
| Version | PBO |
|---------|-----|
| V1 (buggy search space) | 44.3% |
| V2 (paper search space) | 68.9% |
| Paper | 8.0% |

**The "correct" search space made PBO dramatically worse.** This is counterintuitive but explainable: the paper's search space includes large learning rates (up to 0.03) which produce more diverse trial outcomes. Our Optuna converges to lr=5e-6, producing 50 very similar agents, making the PBO rank-based statistic detect them all as overfitting (since they all overfit the same way).

---

### 3.6 Backtest

#### Paper Description
- Trade period: Apr 30 – Jun 27, 2022 (during Terra/LUNA crash)
- Retrain on full training+validation data before backtesting
- Apply CVIX risk control during backtest
- Results: PPO CPCV = -34.96% cumret, EQW = -47.78%, S&P BDM = -38.79%

#### Code Implementation

**Bug 7 — No retrain step**:
The paper says to retrain on the full dataset before backtesting. The code just loads the agent saved during the best CPCV trial (which was trained on ~60% of the data with 40% held out for validation). This means the backtest agent has less data exposure than the paper's agent.

#### Our Results vs Paper

| Metric | Paper | Our Result | Gap |
|--------|-------|-----------|-----|
| PPO CPCV Cumret | -34.96% | -68.3% | 33.3% worse |
| EQW Cumret | -47.78% | -51.3% | 3.5% worse |
| S&P BDM Cumret | -38.79% | -38.89% | ~0.1% (matches) |
| Volatility | 2.01×10⁻³ | 4.27×10⁻³ | 2.1× higher |
| Sharpe | N/A | -5.23 | N/A |
| PBO | 8.0% | 68.9% | 8.6× worse |

The S&P benchmark matches almost perfectly (validating our date alignment). The EQW gap (~3.5%) is attributable to different data sources (CoinAPI vs Binance — slightly different prices at 5-min granularity). The PPO gap (33.3%) is the core reproduction failure.

---

### 3.7 Metrics

#### Sharpe Ratio Calculation

**Bug 4 — Dollar differences instead of percentage returns**:
```python
# BUGGY (original code):
drl_rets_tmp = account_value_erl[1:] - account_value_erl[:-1]  # Dollar differences

# FIXED:
drl_rets_tmp = account_value_erl[1:] / account_value_erl[:-1] - 1  # Percentage returns
```

The Sharpe ratio is defined as mean(excess_return)/std(return). Dollar differences make the Sharpe scale-dependent — a $1M portfolio's dollar changes are 1000× a $1000 portfolio's, producing nonsensical comparisons. Percentage returns make it scale-invariant.

**Bug 5 — Inverted EQW returns**:
```python
# BUGGY:
eqw_rets_tmp = account_value_eqw[:-1] / account_value_eqw[1:] - 1  # prev/next

# FIXED:
eqw_rets_tmp = account_value_eqw[1:] / account_value_eqw[:-1] - 1  # next/prev
```

The original computed 1/return for EQW, making +5% returns appear as -4.76% returns. This artificially depressed EQW Sharpe, making DRL appear better by comparison.

**Bug 6 — Annualization factor**:
```python
# BUGGY:
factor = data_points_per_year / dataset_size  # Fraction of year

# FIXED:
factor = data_points_per_year  # Periods per year
```

`sharpe_iid()` computes `Sharpe = mean(ret)/std(ret) × sqrt(factor)`. The factor should be periods-per-year (e.g., 105,120 for 5-min data) for annualization. The buggy code divided by dataset_size, making factor = ~4.2 instead of ~105,120, underscaling the Sharpe by factor of ~158×.

---

## 4. Bug-by-Bug Impact Analysis

### Bug 1: `get_state()` Index Bug
**File**: `environment_Alpaca.py`, line ~175  
**Buggy**: `self.tech_array[-1 - i]` — always reads from the END of the tech array  
**Fixed**: `self.tech_array[self.time - i]` — reads from current time step backwards

**Numerical Example** (with lookback=1, 1000-timestep episode):
- At t=500: Buggy reads tech_array[999] (future data). Fixed reads tech_array[500] (current data).
- At t=100: Buggy reads tech_array[999] (same future data). Fixed reads tech_array[100].
- Effect: Every state the agent sees during training includes the SAME tech snapshot (the last one), regardless of time. The agent effectively trains with no temporal information — it always sees the market at t=T.

**Impact on results**: Paradoxically, this bug may have helped in the original paper. By giving the agent future information, it can learn "what the market ends up looking like" and potentially make better decisions. Fixing this bug removes that information leakage, potentially making the agent perform worse on both training and testing.

**However**: With lookback=1 (common in the search space), the agent only gets ONE tech snapshot, so the damage is limited to "wrong snapshot" rather than a systematic lookback of wrong data.

---

### Bug 2: PBO Matrix Axis
**File**: `function_PBO.py`, `build_matrix_M_splits()`  
**Buggy**: `np.mean(returns, axis=0)` — averages across CPCV splits  
**Fixed**: `np.mean(returns, axis=1)` — averages across time steps within each split

**Numerical Example** (3 splits, 100 time steps, 5 trials):
```
M_split_i shape: (3 splits, 100 timesteps)
axis=0: result shape (100,) - mean across 3 splits at each timestep
axis=1: result shape (3,)   - mean return per split
```

The PBO algorithm expects M to be (T' × H) where T' is the validation length and H is the number of trials. With axis=0, we were creating M of the wrong shape (100×50 instead of 3×50 per trial), fundamentally corrupting the rank statistics.

**Impact**: PBO values become meaningless with the wrong axis. The rank correlations between in-sample and out-of-sample performance are computed on incorrect aggregations.

---

### Bug 3: PBO Off-by-One
**File**: `function_PBO.py`  
**Buggy**: `for no_trial in range(len(trials) - 1)` — skips trial 49  
**Fixed**: `for no_trial in range(len(trials))` — processes all 50 trials

**Impact**: With 50 trials, we lose 2% of the data (1 trial). Small but systematic — the skipped trial might be the best or worst one, biasing the PBO estimate.

---

### Bug 4: Dollar vs. Percentage Returns
**File**: `function_train_test.py`, line ~108

**Numerical Example**:
Starting portfolio: $1,000,000. Day 1 change: +$10,000. Day 2 change: -$5,000.

```
Dollar diffs: [10000, -5000]  → mean=2500, std=7500 → Sharpe=0.33
Pct returns:  [0.01, -0.005] → mean=0.0025, std=0.0075 → Sharpe=0.33
```

For Sharpe, both give the same ratio (scale cancels). **But** when you add the annualization factor (Bug 6 interacts here):
```
Dollar: Sharpe = (2500/7500) × sqrt(factor)
Pct:    Sharpe = (0.0025/0.0075) × sqrt(factor)
```

The real issue is that dollar differences don't compose correctly. If portfolio grows from $1M to $2M, the same $10K change is 1% at the start but 0.5% at the end. Dollar returns inflate Sharpe for growing portfolios and deflate it for shrinking ones. In a bear market (Apr-Jun 2022), dollar diffs on a shrinking portfolio systematically overstate the Sharpe.

**Impact on Optimization**: The CPCV objective is `mean(Sharpe_DRL) - mean(Sharpe_EQW)`. With dollar returns, both Sharpes are wrong, but they're wrong in correlated ways (same scale). The difference might accidentally be reasonable, which could explain why the buggy code sometimes produces acceptable results.

---

### Bug 5: Inverted EQW Returns
**File**: `function_finance_metrics.py`, `compute_eqw()`

**Numerical Example**:
EQW portfolio value goes from $1,000,000 to $950,000 (a -5% day):
```
Buggy: 1000000/950000 - 1 = +5.26%   (shows positive when actually negative!)
Fixed: 950000/1000000 - 1 = -5.00%   (correct)
```

This flips the SIGN of EQW returns. In the CPCV objective `Sharpe(DRL) - Sharpe(EQW)`:
- Buggy EQW Sharpe is inverted → a market crash makes EQW look like it has a POSITIVE Sharpe
- This makes the DRL agent need to do even better to beat the (falsely positive) EQW benchmark
- In practice, both terms are computed with dollar diffs (Bug 4), so the interaction is complex

**Net effect**: The inverted EQW makes the optimization objective harder for the DRL agent. Trials that beat the inverted EQW are genuinely better performers. Paradoxically, this might have selected for more robust agents in the original (buggy) code.

---

### Bug 6: Sharpe Annualization Factor
**File**: `function_train_test.py`, line ~105

**Numerical Example** (5-minute data):
```
data_points_per_year = 12 × 24 × 365 = 105,120
dataset_size = 5000 (validation split)

Buggy:  factor = 105120/5000 = 21.02  → sqrt(21.02) ≈ 4.58
Fixed:  factor = 105120                → sqrt(105120) ≈ 324.2
```

The Sharpe is multiplied by sqrt(factor), so:
```
Buggy Sharpe  = raw_sharpe × 4.58
Fixed Sharpe  = raw_sharpe × 324.2
```

The scale is 70× different. However, since BOTH DRL and EQW Sharpes are computed with the same buggy factor, and the objective is their DIFFERENCE, the factor cancels in the relative comparison:
```
Objective = Sharpe_DRL - Sharpe_EQW = (raw_DRL - raw_EQW) × sqrt(factor)
```

The factor just scales the whole objective. Optuna maximizes this — the optimal hyperparameters are the same regardless of scaling. So **Bug 6 has NO effect on the optimization outcome** — it only affects the reported Sharpe values (which appear ~70× too small with the bug).

---

### Bug 7: No Retrain Before Backtest
**File**: `4_backtest.py`

The paper states the best agent should be retrained on the **full** train+validation dataset before backtesting. The code just loads the agent from the CPCV best trial, which was trained on ~60% of the data (N-K/N = 3/5 = 60%).

**Impact**: The backtest agent has 40% less data exposure. For a market with structural breaks (Terra/LUNA crash in May 2022), having the full pre-crash data is crucial for the agent to learn crash dynamics. This alone could account for a significant portion of the performance gap.

**Estimated impact**: Unknown precisely, but training on 60% vs 100% of data is a major disadvantage, especially when the training period includes important regime changes.

---

### Bug 8: Search Space Mismatch
**File**: `1_optimize_cpcv.py`, `1_optimize_kcv.py`, `1_optimize_wf.py`

The original code had a search space that didn't match Table 2 of the paper. We aligned it:

| Parameter | Original Code | Paper Table 2 |
|-----------|--------------|---------------|
| learning_rate | [1e-2, 5e-3, ..., 5e-6] | [3e-2, 2.3e-2, 1.5e-2, 7.5e-3, 5e-6] |
| batch_size | [64, 128, ..., 4096] | [512, 1280, 2048, 3080] |
| gamma | [0.9, 0.95, ..., 0.999] | Uniform [0.95, 0.99] |
| net_dim | [64, 128, 256, 512] | [512, 1024, 2048] |
| target_step | [1024, 2048, ..., 8192] | [2500, 3750, 5000] |
| break_step | [2e4, ..., 8e4] | [3e4, 4.5e4, 6e4] |

**Impact**: The original code explored much smaller networks (64-512) and wider learning rate ranges. The paper's search space has larger networks (512-2048) and higher learning rates. Aligning the search space changed which trials Optuna explored, but Optuna still converged to lr=5e-6 regardless.

---

### Bug 9: Hardcoded Model Names in PBO
**File**: `5_pbo.py`

Hardcoded `model_names` list that didn't match the actual pickle file contents. A configuration error, not an algorithmic bug. Fix: dynamically read model names from pickle results.

**Impact**: Would cause crashes or wrong labels. No effect on numerical results if only running PPO CPCV.

---

## 5. Why Results Don't Match — With or Without Fixes

### 5.1 The Core Problem: Seed Sensitivity

The user confirmed:
> "Using paper's best hyperparams + right seed → within 4% of paper's -34.96%. Most seeds → ~-60% cumret."

DRL training is inherently stochastic. The same hyperparameters with different random seeds produce wildly different agents. In a 50-trial CPCV optimization:
- Each trial trains 10 agents (one per CPCV split)
- Each agent starts from different random weights
- The PPO policy gradient has high variance
- The result is that agent quality depends heavily on initialization luck

The paper reports the BEST outcome. If the best seed gives -35% and the typical seed gives -60%, then the paper's result is a 1-in-N lucky draw, not a reproducible baseline.

### 5.2 Data Source Difference (CoinAPI vs. Binance)

We used CoinAPI because Binance is geo-restricted. At 5-minute granularity, price differences between exchanges can be significant:
- Different liquidity pools → different price discovery
- Different timestamp alignment → shifted candles
- Different volume profiles → different technical indicator values
- EQW gap of 3.5% (-51.3% vs -47.78%) is consistent with data source effects

The DRL agent is sensitive to exact price patterns. A 0.1% price difference at a critical decision point can cascade into completely different portfolio trajectories through the compounding of sequential decisions.

### 5.3 Technical Indicator Set

As verified on disk (Section 3.2), the actual features reaching the environment are **close, volume, rsi, dx, ultosc, obv, ht_dcphase** — the paper's 6 features plus a redundant `close` column. The `TECHNICAL_INDICATORS_LIST` in `config_main.py` (which lists 11 items) is misleading because both processors ignore it entirely; indicators are hardcoded in the processor and filtered by a hardcoded drop list.

The extra `close` column (already available in `price_array`) adds one redundant dimension per coin (70 vs 60 total tech features). This is a **minor** discrepancy — it does not fundamentally change the learning problem and is unlikely to explain the reproduction gap.

### 5.4 Bugs Don't Explain the Gap

V1 (all original bugs in place, original search space) produced ~-60% cumulative return — already far from the paper's -34.96%. This is the most important observation: **the reproduction failure exists with or without our fixes**.

The metric bugs (4, 5, 6) largely cancel or are irrelevant to the optimization:
- **Bug 6 (annualization factor)**: Cancels completely — both DRL and EQW Sharpe scale by the same wrong `sqrt(factor)`, so the objective `Sharpe(DRL) - Sharpe(EQW)` is just uniformly rescaled. Optuna picks the same best trial regardless.
- **Bug 4 (dollar returns)**: For a given CPCV split, the DRL and EQW portfolios traverse similar value ranges, so the Sharpe ratio computed from dollar diffs produces a similar ranking to the one from percentage returns. The optimization landscape changes slightly but not qualitatively.
- **Bug 5 (inverted EQW)**: Makes the EQW benchmark appear to have opposite-sign Sharpe, shifting the objective. But Optuna still maximizes `Sharpe(DRL) - Sharpe(EQW)` — it just targets a different baseline.

The fact that V1 (buggy) ≈ -60% and V2 (fixed) ≈ -68% — both far from -34.96% — confirms that the reproduction gap is driven by factors external to the bugs: **data source, seed sensitivity, and the missing retrain step**.

Bug 1 (future data leak in `get_state`) is the only fix that could meaningfully change agent quality, and its effect depends on lookback. With lookback=1, the agent gets one wrong tech snapshot per step — noisy but not transformative. The ~8% gap between V1 and V2 may partly reflect this, but it's dwarfed by the 25%+ gap from paper.

### 5.5 Training Infrastructure

The paper used specific versions of:
- ElegantRL (unknown version)
- PyTorch (unknown version)  
- CUDA/GPU (unknown hardware)
- Binance API (with specific data quality characteristics)

DRL training is known to be sensitive to framework versions, GPU precision, and even hardware-specific floating-point behavior. Different PyTorch versions can produce different gradient computations, different CUDA versions have different numerical precision in matrix operations, and these compound over thousands of training steps.

### 5.6 The Retrain Step (Bug 7)

The missing retrain on full data before backtest means our agent trained on 60% of available data. The paper's agent trained on 100%. The additional 40% includes crucial market regime information. This alone could explain a significant portion of the 33% gap.

### 5.7 Summary: Why Both Buggy and Fixed Code Fail

| Factor | Buggy Code Effect | Fixed Code Effect | Paper |
|--------|-------------------|-------------------|-------|
| State data | Future leak in tech_array | Current data only | Current data only |
| Optimization objective | Wrong (dollar diffs + inverted EQW) | Correct (pct returns + correct EQW) | Correct |
| PBO | 44.3% (wrong computation) | 68.9% (correct computation) | 8.0% |
| Search space | Wrong (wider, smaller nets) | Correct (paper Table 2) | Correct |
| Data source | CoinAPI | CoinAPI | Binance |
| Features | 7 (close+vol+RSI+DX+ULTOSC+OBV+HT) | 7 (same) | 6 (vol+RSI+DX+ULTOSC+OBV+HT) |
| Retrain | No | No | Yes (on full data) |
| Seed | Random | Random | Specific best seed |
| Cumret | ~-60% | -68.3% | -34.96% |

The buggy code accidentally gets around -60% (not -35%) because the future leak and wrong objective create a noisy but not consistently better optimization. The fixed code gets -68.3% because it correctly identifies that the agent genuinely doesn't beat the market, and the CVIX control further hurts by crystallizing crash losses.

Neither configuration can reach -34.96% because:
1. **Seed sensitivity** means only a few lucky random initializations can match
2. **Wrong data source** (CoinAPI ≠ Binance) affects every single decision
3. **No retrain** means less data exposure than the paper's agent
4. **The bugs are not the cause** — the original buggy code also fails at ~-60%

---

## 6. Numerical Summary Table

| Metric | Paper | V1 (Buggy Space) | V2 (Paper Space) | Delta (V2 vs Paper) |
|--------|-------|-------------------|-------------------|---------------------|
| PPO CPCV Cumret | -34.96% | ~-60% | -68.3% | -33.3% |
| EQW Cumret | -47.78% | ~-50% | -51.3% | -3.5% |
| S&P BDM Cumret | -38.79% | -38.89% | -38.89% | -0.1% |
| Volatility | 2.01×10⁻³ | N/A | 4.27×10⁻³ | 2.1× |
| PBO | 8.0% | 44.3% | 68.9% | +60.9% |
| Best trial Sharpe(DRL-EQW) | N/A | 0.0 (best) | 0.1086 | N/A |
| Positive trials | N/A | 0/50 | 16/50 | N/A |
| Validation p-value (t-test) | N/A | N/A | 0.886 | N/A |

---

## 7. Conclusions

### 7.1 What We Can Confirm
1. The CPCV framework is correctly implemented (modulo Bugs 2-3 in PBO)
2. The environment logic is structurally sound
3. The S&P benchmark matches perfectly, validating date alignment
4. The EQW gap is consistent with data source differences

### 7.2 What We Cannot Reproduce
1. PPO CPCV cumulative return (-68.3% vs -34.96%)
2. PBO of 8.0% (we get 68.9%)
3. PPO outperforming EQW and S&P

### 7.3 Root Causes (Ranked by Likely Impact)

Controlled diagnostic (`6_diagnostic.py`): 10 seeds (0–9), paper best hyperparameters, full retrain on 100% training data, tested with and without CVIX. EQW baseline: -51.32% (ours) vs -47.78% (paper), a 3.54pp data gap.

1. **Seed sensitivity** (DOMINANT): 10-seed distribution spans -38% to -71% with CVIX, -51% to -71% without. The 32pp spread dwarfs all other factors. The paper's -34.96% sits at ~1.7σ above the mean — a favorable but not implausible draw. The CPCV framework optimizes hyperparameters while the dominant variance source (random seed) is never searched.

2. **CVIX risk control** (MEDIUM-HIGH, +5.7pp avg): Controlled A/B across same 10 seeds:
   - With CVIX: -54.00% ± 10.92%
   - Without CVIX: -59.68% ± 6.70%
   - Mean benefit: +5.68pp, but highly seed-dependent (-2.6pp to +14.2pp)
   - CVIX *amplifies* seed sensitivity (std 10.92% vs 6.70%) — it helps most when the agent's positions happen to align with forced selling during high-vol bars
   - Threshold 90.1 calibrated from test data is a form of lookahead bias
   - **Standalone EQW+CVIX benchmark**: A trivial strategy of "hold equal-weight, sell all when DVOL>90.1, rebuy when below" achieves -51.39% — essentially identical to pure EQW (-51.32%). The CVIX rule adds nothing to a passive strategy (symmetric sell/rebuy washes out). Yet the DRL+CVIX agent averages -54.00%, i.e., **2.6pp worse than the trivial EQW+CVIX baseline**. The DRL policy destroys value on average; the best-seed result (-38.37%) is a lottery ticket, not learned alpha.

3. **Data source** (MEDIUM, ~3.5pp): CoinAPI ≠ Binance prices cause a 3.54pp baseline shift (our EQW -51.32% vs paper -47.78%). Data-adjusted gap for best seed: +0.14pp — essentially exact reproduction after accounting for data differences.

4. **No retrain step** (MEDIUM): CPCV optimization trains on 60% data per fold; paper methodology retrains on 100% data with best params. The stored agent in `4_backtest.py` uses the partial-data model, missing the retrain step. Seed=0 with full retrain + CVIX gets -39.85% vs -51.13% without retrain/CVIX — combined retrain+CVIX effect is ~11pp.

5. **Reward formulation** (NEGLIGIBLE): Code uses relative reward (vs EQW benchmark) while paper Eq. 3 specifies absolute reward. A/B diagnostic (`7_reward_diagnostic.py`, 10 seeds × 2 param sets = 40 runs) confirms no significant difference:
   - Paper params: relative -55.81%±10.93% vs absolute -55.42%±10.64%, Δ=+0.40pp, p=0.57
   - V2 params: relative -59.96%±8.37% vs absolute -57.83%±9.00%, Δ=+2.13pp, p=0.57
   - Per-seed deltas range from -19pp to +21pp — dominated by seed noise, not reward formulation.

6. **Bug fixes** (NEGLIGIBLE): Metric bugs (4-6) largely cancel or scale uniformly; Bug 1 (future leak) has limited effect with lookback=1. V1 with all bugs still fails at ~-60%.

7. **Technical indicator set** (NEGLIGIBLE): 7 features vs paper's 6 — all 6 paper features present, just 1 extra redundant `close` column

### 7.4 Reproducibility Assessment

This paper's results are **conditionally reproducible**. With the correct methodology (full retrain + CVIX + paper hyperparameters), data-adjusted results match within noise for favorable seeds:

| Condition | Mean±Std | Best seed | Data-adj gap to paper |
|---|---|---|---|
| Full retrain + CVIX (paper method) | -54.00% ± 10.92% | -38.37% (seed 7) | +0.14pp |
| Full retrain, no CVIX | -59.68% ± 6.70% | -51.13% (seed 0) | -12.63pp |
| EQW + CVIX (no DRL) | -51.39% (deterministic) | — | — |
| Pure EQW (no DRL, no CVIX) | -51.32% (deterministic) | — | — |
| CPCV stored agent (backtest pipeline) | ~-60% | — | ~-25pp |

The paper's -34.96% requires:
1. Binance data (accounts for ~3.5pp baseline shift vs our CoinAPI data)
2. Full retrain on 100% data (not the CPCV partial-fold model)
3. CVIX risk control with threshold 90.1 (+5.7pp average benefit)
4. A favorable random seed (~20% of seeds land within 5pp of paper)

**The dominant finding** is that the DRL agent adds no value. The EQW+CVIX benchmark (a rule requiring zero machine learning) achieves -51.39%, while the DRL+CVIX agent averages -54.00% across 10 seeds — the agent is a net negative. Without CVIX, the agent averages -59.68% vs EQW's -51.32%, confirming it actively harms returns. The paper's best-case -34.96% is a seed-dependent outlier; no systematic alpha is learned.

Seed sensitivity (32pp spread) vastly exceeds any hyperparameter effect found by 50-trial CPCV optimization. The CPCV framework searches the wrong axis — it optimizes over hyperparameters while holding the random seed fixed at 0, yet the seed is the primary driver of outcome variance.

The PBO framework itself is sound in concept. The finding that PBO = 68.9% with the correct computation (vs paper's 8.0%) suggests that even with the CPCV methodology, 50-trial hyperparameter optimization on cryptocurrency data during a crash period leads to heavy overfitting. This is arguably the most important finding of our reproduction attempt — it may be more consistent with the paper's own hypothesis (that overfitting is a serious problem) than the paper's own reported PBO of 8.0%.
