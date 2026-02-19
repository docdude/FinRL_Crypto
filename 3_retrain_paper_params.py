"""
Retrain PPO agent using the exact hyperparameters published in the paper
(Table 2 of arXiv:2209.05559):

    Learning rate:  7.5e-3
    Batch size:     512
    Gamma:          0.95
    Net dimension:  1024
    Target step:    5e3 (typo in paper says 5e4, but code should uses 5e3, to align with search params)
    Break step:     4.5e4

Trains on the full train+val dataset (25,000 candles) and saves the agent
to train_results/<result_dir>/retrained_agent_paper_params/
"""

import os
import sys
import pickle
from distutils.dir_util import copy_tree

from config_main import (
    TIMEFRAME,
    no_candles_for_train,
    no_candles_for_val,
)
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl


# ── Paper Table 2: PPO hyperparameters ──────────────────────────────────
ERL_PARAMS = {
    "learning_rate": 7.5e-3,
    "batch_size": 512,
    "gamma": 0.95,
    "net_dimension": 1024,
    "target_step": int(5e3),
    "eval_time_gap": 60,
    "break_step": int(4.5e4),
}

# Environment normalization (same single-value search space used in code)
ENV_PARAMS = {
    "lookback": 1,
    "norm_cash": 2 ** -12,
    "norm_stocks": 2 ** -8,
    "norm_tech": 2 ** -15,
    "norm_reward": 2 ** -10,
    "norm_action": 10000,
}

MODEL_NAME = "ppo"


def load_full_data():
    data_folder = f'./data/{TIMEFRAME}_{no_candles_for_train + no_candles_for_val}'
    print(f'\nLoading full dataset from: {data_folder}\n')
    with open(f'{data_folder}/price_array', 'rb') as f:
        price_array = pickle.load(f)
    with open(f'{data_folder}/tech_array', 'rb') as f:
        tech_array = pickle.load(f)
    with open(f'{data_folder}/time_array', 'rb') as f:
        time_array = pickle.load(f)
    return price_array, tech_array, time_array


def main(result_dir):
    price_array, tech_array, time_array = load_full_data()
    total_candles = price_array.shape[0]
    print(f'Total candles: {total_candles}')
    print(f'Tickers:       {price_array.shape[1]}')
    print(f'Tech features: {tech_array.shape[1]}\n')

    # Paths
    cwd_train = f'./train_results/cwd_tests/retrain_paper_params'
    save_dir = f'./train_results/{result_dir}/retrained_agent_paper_params/'
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 70)
    print('RETRAINING PPO WITH PAPER HYPERPARAMETERS (Table 2)')
    print('=' * 70)
    for k, v in ERL_PARAMS.items():
        print(f'  {k:20s}: {v}')
    for k, v in ENV_PARAMS.items():
        print(f'  {k:20s}: {v}')
    print('=' * 70 + '\n')

    agent = DRLAgent_erl(
        env=CryptoEnvAlpaca,
        price_array=price_array,
        tech_array=tech_array,
        env_params=ENV_PARAMS,
        if_log=True,
    )

    model = agent.get_model(
        MODEL_NAME,
        gpu_id=0,
        model_kwargs=ERL_PARAMS,
    )

    agent.train_model(
        model=model,
        cwd=cwd_train,
        total_timesteps=ERL_PARAMS["break_step"],
    )

    # Copy trained agent
    print(f'\nCopying agent to: {save_dir}')
    copy_tree(cwd_train, save_dir)

    # Save a best_trial-like pickle so 4_backtest.py can load it
    trial_info = {
        'model_name': MODEL_NAME,
        'params': {**ERL_PARAMS, **ENV_PARAMS},
        'user_attrs': {
            'model_name': MODEL_NAME,
            'timeframe': TIMEFRAME,
            'source': 'paper_table2',
        },
    }
    with open(f'./train_results/{result_dir}/paper_params_info', 'wb') as f:
        pickle.dump(trial_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n' + '=' * 70)
    print('RETRAIN COMPLETE — paper params')
    print(f'Agent saved to: {save_dir}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        results_root = './train_results'
        available = sorted([d for d in os.listdir(results_root)
                            if os.path.isdir(os.path.join(results_root, d))
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

    main(result_dir)
