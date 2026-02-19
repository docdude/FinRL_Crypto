"""
Step 3 — Re-train on full training data.

The paper (Section "Training a Trading Agent") says:
    "Loop for H trials and select the set of hyperparameters that performs
    the best... Then, pick the DRL agent with the best-performing
    hyperparameters and *re-train the agent on the whole training data*."

The optimization scripts (1_optimize_*.py) train on CV subsets (e.g., 3/5
of the data for CPCV). This script takes the winning hyperparameters and
retrains on ALL train+val candles so the final backtested agent has seen
the full training period.

Usage:
    python 3_retrain.py

    When prompted, enter the result folder name from train_results/
    e.g.: res_2026-02-11__06_18_08_model_CPCV_ppo_5m_50H_25k
"""

import os
import sys
import pickle
from distutils.dir_util import copy_tree

from config_main import (
    TIMEFRAME,
    no_candles_for_train,
    no_candles_for_val,
    TICKER_LIST,
)
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl


def load_best_trial(result_dir):
    """Load the best trial pickle from a completed optimization run."""
    trial_path = f'./train_results/{result_dir}/best_trial'
    if not os.path.exists(trial_path):
        raise FileNotFoundError(
            f"No best_trial file found in train_results/{result_dir}/\n"
            f"Make sure optimization (step 1) and validation (step 2) completed."
        )
    with open(trial_path, 'rb') as f:
        best_trial = pickle.load(f)
    return best_trial


def load_full_data():
    """Load the full train+val dataset (all candles)."""
    data_folder = f'./data/{TIMEFRAME}_{no_candles_for_train + no_candles_for_val}'
    print(f'\nLoading full dataset from: {data_folder}\n')

    with open(f'{data_folder}/price_array', 'rb') as f:
        price_array = pickle.load(f)
    with open(f'{data_folder}/tech_array', 'rb') as f:
        tech_array = pickle.load(f)
    with open(f'{data_folder}/time_array', 'rb') as f:
        time_array = pickle.load(f)
    return price_array, tech_array, time_array


def extract_hyperparams(best_trial):
    """Extract erl_params and env_params from the best trial."""
    erl_params = {
        "learning_rate": best_trial.params['learning_rate'],
        "batch_size": best_trial.params['batch_size'],
        "gamma": best_trial.params['gamma'],
        "net_dimension": best_trial.params['net_dimension'],
        "target_step": best_trial.params['target_step'],
        "eval_time_gap": best_trial.params['eval_time_gap'],
        "break_step": best_trial.params['break_step'],
    }
    env_params = {
        "lookback": best_trial.params['lookback'],
        "norm_cash": best_trial.params['norm_cash'],
        "norm_stocks": best_trial.params['norm_stocks'],
        "norm_tech": best_trial.params['norm_tech'],
        "norm_reward": best_trial.params['norm_reward'],
        "norm_action": best_trial.params['norm_action'],
    }
    return erl_params, env_params


def retrain_on_full_data(result_dir):
    """Main retrain logic."""
    # Load best trial
    best_trial = load_best_trial(result_dir)
    model_name = best_trial.user_attrs['model_name']
    print(f'Best trial: #{best_trial.number}')
    print(f'Model:      {model_name}')
    print(f'Params:     {best_trial.params}\n')

    # Extract hyperparameters
    erl_params, env_params = extract_hyperparams(best_trial)

    # Load full dataset
    price_array, tech_array, time_array = load_full_data()
    total_candles = price_array.shape[0]
    print(f'Total candles: {total_candles}')
    print(f'Tickers:       {price_array.shape[1]}')
    print(f'Tech features: {tech_array.shape[1]}\n')

    # Use ALL candles for training (the full train+val period)
    train_indices = list(range(total_candles))
    print(f'Training on {len(train_indices)} candles (full dataset)\n')

    # Set up paths
    cwd_train = f'./train_results/cwd_tests/retrain_{result_dir}'
    retrain_dir = f'./train_results/{result_dir}/retrained_agent/'
    os.makedirs(retrain_dir, exist_ok=True)

    # Slice arrays for training
    price_array_train = price_array[train_indices, :]
    tech_array_train = tech_array[train_indices, :]

    # Train
    env = CryptoEnvAlpaca
    break_step = erl_params['break_step']

    print('=' * 70)
    print(f'RETRAINING {model_name.upper()} on FULL data ({len(train_indices)} candles)')
    print(f'break_step = {break_step}')
    print('=' * 70 + '\n')

    agent = DRLAgent_erl(
        env=env,
        price_array=price_array_train,
        tech_array=tech_array_train,
        env_params=env_params,
        if_log=True,
    )

    model = agent.get_model(
        model_name,
        gpu_id=0,
        model_kwargs=erl_params,
    )

    agent.train_model(
        model=model,
        cwd=cwd_train,
        total_timesteps=break_step,
    )

    # Copy retrained agent to result folder
    print(f'\nCopying retrained agent to: {retrain_dir}')
    copy_tree(cwd_train, retrain_dir)

    # Update the best_trial pickle with retrain info
    best_trial.set_user_attr('retrained', True)
    best_trial.set_user_attr('retrained_candles', len(train_indices))
    with open(f'./train_results/{result_dir}/best_trial', 'wb') as f:
        pickle.dump(best_trial, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n' + '=' * 70)
    print('RETRAIN COMPLETE')
    print(f'Agent saved to: {retrain_dir}')
    print('=' * 70)
    print('\nNow update 4_backtest.py pickle_results list and change')
    print("the agent path from 'stored_agent/' to 'retrained_agent/'")
    print('to backtest with the retrained agent.\n')
    return retrain_dir


if __name__ == '__main__':
    # Accept CLI argument or interactive selection
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        results_dir = './train_results'
        available = sorted([d for d in os.listdir(results_dir)
                     if os.path.isdir(os.path.join(results_dir, d))
                     and d.startswith('res_')])
        if available:
            print('\nAvailable result folders:')
            for i, d in enumerate(available):
                print(f'  [{i}] {d}')
            print()
            idx = int(input('Select index: '))
            result_dir = available[idx]
        else:
            result_dir = input('Enter result folder name: ').strip()
    retrain_on_full_data(result_dir)
