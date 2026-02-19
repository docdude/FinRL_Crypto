"""
This script contains a set of functions for loading and processing data for trading.

It contains the following functions:

load_validated_model: Loads the best trial from the pickle file in the specified directory and returns the best trial's attributes.
download_CVIX: Downloads the CVIX dataframe from Yahoo finance and returns it.
load_and_process_data: loads and process the trade data from the specified data folder and returns the data.

After that, the large loop analyzes every result by creating an instance of an Alpaca environment and checking
what the model would do through the environment using the new trading data

Finally, the resulting backtests are analyzes for performance a performance metric per benchmark (EQW, S&P BCI) plus
all the input DRL agents are analyzed.

"""


import os
import pickle
import matplotlib.dates as mdates

from config_main import *
from function_finance_metrics import *
from processor_Yahoo import Yahoofinance
from environment_Alpaca import CryptoEnvAlpaca
#from environment_CCXT import CryptoEnvCCXT
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl


def load_validated_model(result):

    with open('./train_results/' + result + '/best_trial', 'rb') as handle:
        best_trial = pickle.load(handle)

    print('BEST TRIAL: ', best_trial.number)
    timeframe = best_trial.user_attrs['timeframe']
    ticker_list = best_trial.user_attrs['ticker_list']
    technical_ind = best_trial.user_attrs['technical_indicator_list']
    net_dim = best_trial.params['net_dimension']
    model_name = best_trial.user_attrs['model_name']

    print('\nMODEL_NAME: ', model_name)
    print(best_trial.params)
    print(timeframe)

    name_test = best_trial.user_attrs['name_test']

    env_params = {
        "lookback": best_trial.params['lookback'],
        "norm_cash": best_trial.params['norm_cash'],
        "norm_stocks": best_trial.params['norm_stocks'],
        "norm_tech": best_trial.params['norm_tech'],
        "norm_reward": best_trial.params['norm_reward'],
        "norm_action": best_trial.params['norm_action']
    }
    return env_params, net_dim, timeframe, ticker_list, technical_ind, name_test, model_name


def download_CVIX(trade_start_date, trade_end_date):
    """Download crypto volatility index data.

    Primary: Deribit DVOL (BTC implied volatility index) — free public API.
    Fallback: Yahoo CVOL-USD (delisted as of ~2024).
    """
    trade_start = trade_start_date[:10]
    trade_end = trade_end_date[:10]

    # --- Try Deribit DVOL first ---
    try:
        from urllib.request import urlopen, Request
        import json
        from datetime import datetime as dt, timezone

        start_ms = int(dt.strptime(trade_start, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(dt.strptime(trade_end, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

        # Deribit limits to 1000 entries per call, fetch in chunks
        all_entries = []
        chunk_start = start_ms
        while chunk_start < end_ms:
            url = (f'https://www.deribit.com/api/v2/public/get_volatility_index_data'
                   f'?currency=BTC&start_timestamp={chunk_start}'
                   f'&end_timestamp={end_ms}&resolution=3600')
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urlopen(req, timeout=30)
            data = json.loads(resp.read())
            entries = data['result']['data']
            if not entries:
                break
            all_entries.extend(entries)
            # Move past the last entry
            chunk_start = entries[-1][0] + 1

        if not all_entries:
            raise ValueError('Deribit DVOL returned no data')

        # Deduplicate and sort
        seen = set()
        unique = []
        for e in all_entries:
            if e[0] not in seen:
                seen.add(e[0])
                unique.append(e)
        unique.sort(key=lambda x: x[0])

        # Build DataFrame: [timestamp, open, high, low, close]
        dvol_df = pd.DataFrame(unique, columns=['ts', 'open', 'high', 'low', 'close'])
        dvol_df.index = pd.to_datetime(dvol_df['ts'], unit='ms', utc=True)
        dvol_df = dvol_df[['close']]
        dvol_df = dvol_df.resample('5Min').interpolate(method='linear')
        print(f'  DVOL (Deribit): {len(dvol_df)} points, '
              f'range {dvol_df["close"].min():.1f}-{dvol_df["close"].max():.1f}')
        return dvol_df['close']

    except Exception as e:
        print(f'  Deribit DVOL failed: {e}')

    # --- Fallback: Yahoo CVOL-USD ---
    try:
        import yfinance as yf
        df = yf.download('CVOL-USD', start=trade_start, end=trade_end, interval='60m')
        if df.empty:
            raise ValueError('CVOL-USD returned no data (likely delisted)')
        df = df[['Close']].rename(columns={'Close': 'close'})
        df = df.resample('5Min').interpolate(method='linear')
        return df['close']
    except Exception as e:
        print(f'  Yahoo CVOL-USD failed: {e}')

    print('  WARNING: No volatility index available. Backtest will run WITHOUT CVIX risk control.\n')
    return None


def load_and_process_data(TIMEFRAME, trade_start_date, trade_end_date):
    data_folder = f'./data/trade_data/{TIMEFRAME}_{str(trade_start_date[2:10])}_{str(trade_end_date[2:10])}'
    print(f'\nLOADING DATA FOLDER: {data_folder}\n')
    with open(data_folder + '/data_from_processor', 'rb') as handle:
        data_from_processor = pickle.load(handle)
    with open(data_folder + '/price_array', 'rb') as handle:
        price_array = pickle.load(handle)
    with open(data_folder + '/tech_array', 'rb') as handle:
        tech_array = pickle.load(handle)
    with open(data_folder + '/time_array', 'rb') as handle:
        time_array = pickle.load(handle)

    CVIX_series = download_CVIX(trade_start_date, trade_end_date)
    if CVIX_series is not None:
        CVIX_df = pd.merge(time_array.to_series(), CVIX_series, left_index=True, right_index=True, how='left')
        cvix_array = CVIX_df.iloc[:, -1].values
        cvix_array_growth = np.diff(cvix_array)
    else:
        cvix_array = None
        cvix_array_growth = None

    return data_from_processor, price_array, tech_array, time_array, cvix_array, cvix_array_growth


# Inputs
#######################################################################################################
#######################################################################################################
#######################################################################################################

print('TRADE_START_DATE             ', trade_start_date)
print('TRADE_END_DATE               ', trade_end_date, '\n')

pickle_results = ["res_2026-02-18__05_04_45_model_CPCV_ppo_5m_50H_25k",
                  ]

# Execution
#######################################################################################################
#######################################################################################################

drl_cumrets_list = []
model_names_list = []

_, _, timeframe, ticker_list, technical_ind, _, _ = load_validated_model(pickle_results[0])
data_from_processor, price_array, tech_array, time_array, cvix_array, cvix_array_growth = load_and_process_data(TIMEFRAME, trade_start_date, trade_end_date)

for count, result in enumerate(pickle_results):
    env_params, net_dim, timeframe, ticker_list, technical_ind, name_test, model_name = load_validated_model(result)
    model_names_list.append(model_name)

    # Use stored_agent from optimization (original pipeline has no retrain step)
    cwd = './train_results/' + result + '/stored_agent/'
    print(f'  Using stored agent: {cwd}')

    data_config = {
        "cvix_array": cvix_array,
        "cvix_array_growth": cvix_array_growth,
        "time_array": time_array,
        "price_array": price_array,
        "tech_array": tech_array,
        "if_train": False,
    }

    env = CryptoEnvAlpaca
    env_instance = env(config=data_config,
                       env_params=env_params,
                       if_log=True
                       )

    account_value_erl = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dim,
        environment=env_instance,
        gpu_id=0
    )

    # Correct slicing (due to DRL start/end)
    lookback = env_params['lookback']
    indice_start = lookback - 1
    indice_end = len(price_array) - lookback
    time_array = time_array[indice_start:indice_end]

    # Slice cvix array
    if count == 0 and cvix_array is not None:
        cvix_array = cvix_array[indice_start:indice_end]
        cvix_array_growth = cvix_array_growth[indice_start:indice_end]

    # Compute Sharpe's of each coin
    account_value_eqw, ewq_rets, eqw_cumrets = compute_eqw(price_array, indice_start, indice_end)

    # Compute annualization factor
    data_points_per_year = compute_data_points_per_year(timeframe)
    factor = data_points_per_year  # Bug 6 fix: sharpe_iid uses sqrt(factor)

    # Compute DRL rets
    account_value_erl = np.array(account_value_erl)
    drl_rets = account_value_erl[1:] / account_value_erl[:-1] - 1  # Bug 4 fix: percentage returns
    drl_cumrets = [x / account_value_erl[0] - 1 for x in account_value_erl]
    drl_cumrets_list.append(drl_cumrets)

    # Compute metrics per pickle result
    #######################################################################################################

    # Only compute consistent metrics once
    if count == 0:
        # Load S&P index
        spy_index_df = pd.read_csv('data/SPY_Crypto_Broad_Digital_Market_Index - Sheet1.csv')
        spy_index_df['Date'] = pd.to_datetime(spy_index_df['Date'])

        account_value_spy = np.array(spy_index_df['S&P index'])
        spy_rets = account_value_spy[:-1] / account_value_spy[1:] - 1
        spy_rets = np.insert(spy_rets, 0, 0)
        spy_index_df['cumrets_sp_idx'] = [x / spy_index_df['S&P index'][0] - 1 for x in spy_index_df['S&P index']]
        spy_index_df['rets_sp_idx'] = spy_rets
        spy_index_df.set_index('Date', inplace=True)
        spy_index_df = spy_index_df.resample('5Min').interpolate(method='pchip')

        sp_annual_ret, sp_annual_vol, sp_sharpe_rat, sp_vol = aggregate_performance_array(spy_rets, factor)

        write_metrics_to_results('S&P Broad Crypto index',
                                 'plots_and_metrics/test_metrics.txt',
                                 spy_index_df['cumrets_sp_idx'],
                                 sp_annual_ret,
                                 sp_annual_vol,
                                 sp_sharpe_rat,
                                 sp_vol,
                                 'w'
                                 )

        # Write buy-and-hold strategy
        eqw_annual_ret, eqw_annual_vol, eqw_sharpe_rat, eqw_vol = aggregate_performance_array(np.array(ewq_rets),
                                                                                                 factor)
        write_metrics_to_results('Buy-and-Hold',
                                 'plots_and_metrics/test_metrics.txt',
                                 eqw_cumrets,
                                 eqw_annual_ret,
                                 eqw_annual_vol,
                                 eqw_sharpe_rat,
                                 eqw_vol,
                                 'a'
                                 )

    # Then compute the actual metrics from the DRL agents
    drl_annual_ret, drl_annual_vol, drl_sharpe_rat, drl_vol = aggregate_performance_array(np.array(drl_rets), factor)
    write_metrics_to_results(model_name,
                             'plots_and_metrics/test_metrics.txt',
                             drl_cumrets,
                             drl_annual_ret,
                             drl_annual_vol,
                             drl_sharpe_rat,
                             drl_vol,
                             'a'
                             )

    # Hold out of loop only add once
    #######################################################################################################

# Plot
#######################################################################################################
#######################################################################################################

drl_rets_array = np.transpose(np.vstack(drl_cumrets_list))

# General 1
plt.rcParams.update({'font.size': 22})
plt.figure(dpi=300)
f, ax1 = plt.subplots(figsize=(20, 8))

# Plot returns
line_width = 2
ax1.plot(spy_index_df.index, spy_index_df['cumrets_sp_idx'].values, linewidth=3, label='S&P BDM Index')
ax1.plot(time_array, eqw_cumrets[1:], linewidth=line_width, label='Equal-weight', color='blue')


for i in range(np.shape(drl_rets_array)[1]):
    ax1.plot(time_array, drl_rets_array[:, i], label=model_names_list[i], linewidth=line_width)
ax1.legend(frameon=False, ncol=len(model_names_list) + 2, loc='upper left', bbox_to_anchor=(0, 1.11))
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth(3)
ax1.grid()

# Plot CVIX
if cvix_array is not None:
    ax2 = ax1.twinx()
    ax2.plot(time_array, cvix_array, linewidth=4, label='CVIX', color='black', linestyle='dashed', alpha=0.4)
    ax2.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.7, 1.17))
    ax2.patch.set_edgecolor('black')
    ax2.patch.set_linewidth(3)
    ax2.set_ylabel('CVIX')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax1.set_ylabel('Cumulative return')
plt.xlabel('Date')
plt.legend()
plt.savefig('./plots_and_metrics/test_cumulative_return.png', bbox_inches='tight')
if cvix_array is not None:
    ax2.patch.set_edgecolor('black')
    ax2.patch.set_linewidth(3)
    ax2.set_ylabel('CVIX')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax1.set_ylabel('Cumulative return')
plt.xlabel('Date')
plt.legend()
plt.savefig('./plots_and_metrics/test_derivative_CVIX.png', bbox_inches='tight')
