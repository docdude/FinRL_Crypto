# Multiple code bugs prevent reproduction of paper results (arxiv:2209.05559)

## Summary

I attempted a full end-to-end reproduction of the results in *"Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting"* (arxiv:2209.05559) using this repository. After extensive debugging, I found **9 bugs/discrepancies** between the code and the paper. Several are severe enough that key results (particularly the PBO computation, which is the paper's central contribution) could never have been produced by this code.

I'm raising this because the README states *"To reproduce the results in the paper, the codes are simplified as much as possible"*, and the methodology described in the paper is genuinely valuable. However, the code as published does not implement it correctly.

**Note:** All 9 bugs listed below are present in both the original author's repository ([berendgort/FinRL_Crypto](https://github.com/berendgort/FinRL_Crypto)) and the AI4Finance fork ([AI4Finance-Foundation/FinRL_Crypto](https://github.com/AI4Finance-Foundation/FinRL_Crypto)). The AI4Finance fork appears to have been forked without modification of the core logic.

---

## Bug 1 (Critical): `get_state()` reads from wrong timestep

**File:** `environment_Alpaca.py`, `get_state()` method

**Bug:** The state construction uses `self.tech_array[-1 - i]` which always reads technical indicators from the **end** of the array, regardless of the current timestep:

```python
def get_state(self):
    state = np.hstack((self.cash * self.norm_cash, self.stocks * self.norm_stocks))
    for i in range(self.lookback):
        tech_i = self.tech_array[-1 - i]  # BUG: always reads from end of array
        normalized_tech_i = tech_i * self.norm_tech
        state = np.hstack((state, normalized_tech_i)).astype(np.float32)
    return state
```

**Impact:** The agent receives the same (wrong) technical indicator values at every timestep during both training and inference. It never sees the actual current market state.

**Fix:** `self.tech_array[self.time - i]`

---

## Bug 2 (Critical): PBO matrix `axis=0` — the paper's headline metric has never worked

**File:** `5_pbo.py`, `build_matrix_M_splits()` function

**Bug:** `np.mean(rets_single_trial, axis=0)` averages across the **time** dimension instead of across **CPCV splits**:

```python
def build_matrix_M_splits(trials, number_of_trials):
    matrix_cumrets_val = []
    for i in range(number_of_trials):
        trial = trials[i]
        drl_rets_val_list = trial.user_attrs['drl_rets_val_list']
        drl_rets_val_list = add_samples_equify_array_length(drl_rets_val_list)
        rets_single_trial = np.vstack(drl_rets_val_list)
        rets_single_trial = np.mean(rets_single_trial, axis=0)  # BUG: axis=0 averages over TIME
        matrix_cumrets_val.append(rets_single_trial)
    matrix_cumrets_val = np.transpose(np.vstack(matrix_cumrets_val))
    return matrix_cumrets_val
```

**What happens:** After `add_samples_equify_array_length`, the array shape is `(10012, 10)` — 10012 time steps × 10 CPCV splits. `axis=0` collapses the 10012 time steps into a single row, producing M with shape `(10, N_trials)`. With `S=14`, `sub_T = 10 // 14 = 0`, and all CSCV chunks are empty, producing **all NaN logits and PBO = 0.0%**.

**What the paper says (Section "Estimating the Probability of Overfitting"):**
> *Step 1: For each hyperparameter trial, average the returns on the validation sets (of length T') and obtain R_avg ∈ ℝ^{T'}*
> *Step 2: For H trials, stack R_avg into a matrix M ∈ ℝ^{T' × H}*

So M should be `(T' × H) = (10012 × 50)`, not `(10 × 49)`.

**Fix:** `np.mean(rets_single_trial, axis=1)` — average across splits (axis=1), preserving the time dimension.

**Result after fix:** M shape becomes `(10012, 50)`, S=14 yields `sub_T = 715`, and PBO computes correctly with real logit values.

---

## Bug 3: Off-by-one drops last trial from PBO

**File:** `5_pbo.py`, `load_validated_model()` function

```python
number_of_trials = len(trials) - 1  # drops trial index 49 silently
```

With H=50 trials, only trials 0–48 are used in the PBO matrix. The last trial is silently excluded.

---

## Bug 4: Dollar returns fed to Sharpe ratio expecting percentage returns

**File:** `function_train_test.py`, `test_agent()` function

```python
drl_rets_tmp = account_value_erl[1:] - account_value_erl[:-1]  # dollar differences
sharpe_bot, _ = sharpe_iid(drl_rets_tmp, bench=0, factor=factor, log=False)
```

`sharpe_iid` with `log=False` calls `pct_to_log_excess()` → `pct_to_log_return()` which computes `np.log(1 + returns)`. For dollar differences in the hundreds of thousands, this produces meaningless values. Should be percentage returns: `account_value_erl[1:] / account_value_erl[:-1] - 1`.

---

## Bug 5: Equal-weight returns computed backwards

**File:** `function_finance_metrics.py`, `compute_eqw()` function

```python
eqw_rets_tmp = account_value_eqw[:-1] / account_value_eqw[1:] - 1  # divides t by t+1 (inverted)
```

Standard percentage return is `P(t+1) / P(t) - 1`, but this computes `P(t) / P(t+1) - 1`, inverting the sign of returns.

---

## Bug 6: Sharpe annualization factor

**File:** `function_train_test.py`, `test_agent()` function

```python
factor = data_points_per_year / dataset_size
```

With 5-minute data, `data_points_per_year = 105120` and `dataset_size ≈ 10012`, giving `factor ≈ 10.5`. The `sharpe_iid` function applies `np.sqrt(factor)`, which would be correct if factor were the number of periods per year. But dividing by `dataset_size` makes it a ratio that doesn't correspond to standard annualization. This factor should be `data_points_per_year` (or 1 if no annualization is desired for the optimization objective).

---

## Bug 7: No retrain step before backtesting

**Paper (Section "Training a Trading Agent"):**
> *"We pick the DRL agent with the best-performing hyperparameters and re-train the agent on the whole training data."*

**Code:** `4_backtest.py` loads whatever checkpoint was saved during the **last CPCV split** of the best trial. There is no retraining on the full training data. The checkpoint reflects training on only 3 out of 5 groups (60% of data).

---

## Bug 8: Search space mismatch

**Paper Table 1** lists specific hyperparameter values (e.g., learning rate: {7.5e-3, 1.5e-2, 3e-2, 5e-2, 7.5e-2}), but the code uses different ranges. For example, gamma options in the paper are {0.95, 0.96, 0.97,0. 98, 0.99} but the code varies across files. The total combinations also don't match (paper says 2700, code search spaces produce different counts).

---

## Bug 9: Hardcoded plot labels

**File:** `5_pbo.py`

```python
model_names = ['WF', 'KCV', 'CPCV']  # hardcoded, ignores actual model_names
```

This overwrites the dynamically built `model_names` list, so if you run a single CPCV experiment, the plot labels it as "WF".

---

## Reproduction attempt results

The 50-trial CPCV optimization was run using the AI4Finance FinRL_Crypto codebase with only the `get_state` bug (Bug 1) fixed beforehand. Bugs 4–6 (dollar returns, inverted EQW returns, Sharpe factor) remained in the training/evaluation loop as-is from the original repo. Bugs 2, 3, and 9 (PBO axis, off-by-one, plot labels) were identified and fixed post-optimization for the PBO analysis, since they only affect the PBO computation step.

| Metric | Our Result | Paper (PPO CPCV) |
|--------|-----------|------------------|
| Cumulative Return | -66.08% | -34.96% |
| PBO | 44.3% | 8.0% |
| Matrix M shape | (10012, 50) ✓ | ℝ^{T' × H} |

The cumulative return gap is partially attributable to the data source difference (CoinAPI vs. original Binance) and search space mismatch. However, bugs 1 and 2 alone are severe enough that the published code could not have produced the paper's results — Bug 1 means the agent never observes the current market state, and Bug 2 means the PBO computation produces all NaN logits.

---

## Environment

- Python 3.12.3
- Ubuntu 22.04
- Data: CoinAPI (original Binance API no longer serves historical data for the paper's date range without archival access)

## Questions for the authors

1. Was the paper's PBO computed using different code than what's in this repository?
2. Are the correct hyperparameter search spaces available?
3. Was a retrain step performed before backtesting, and if so, is that code available?

Thank you for the interesting paper — the CPCV + PBO methodology for DRL trading is a genuinely useful framework. These issues are raised in the spirit of enabling others to build on it.
