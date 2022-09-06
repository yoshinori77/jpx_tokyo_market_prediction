import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from lightgbm import LGBMRanker, LGBMRegressor
from multiprocessing import cpu_count
from metrics import custom_metric


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2, rank_col='Rank') -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """

    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df[rank_col].min() == 0
        assert df[rank_col].max() == len(df[rank_col]) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by=rank_col)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by=rank_col, ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio, buf


def set_rank(df):
    """
    Args:
        df (pd.DataFrame): including predict column
    Returns:
        df (pd.DataFrame): df with Rank
    """
    # sort records to set Rank
    df = df.sort_values("predict", ascending=False)
    # set Rank starting from 0
    df.loc[:, "Rank"] = np.arange(len(df["predict"]))
    return df


train = pd.read_parquet('../Output/financials_train_agg_df.parquet')
val = pd.read_parquet('../Output/financials_val_agg_df.parquet')
df = pd.concat([train, val]).reset_index(drop=True)
rank_num = 50
df["qcut"] = pd.qcut(df['Target'], rank_num).cat.codes

df["Section/Products"] = df["Section/Products"].astype('category').cat.codes
df["NewMarketSegment"] = df["NewMarketSegment"].astype('category').cat.codes
df["33SectorCode"] = df["33SectorCode"].astype('category').cat.codes
df["17SectorCode"] = df["17SectorCode"].astype('category').cat.codes
df["NewIndexSeriesSizeCode"] = df["NewIndexSeriesSizeCode"].astype('category').cat.codes
df["week"] = df["week"].astype('category').cat.codes
df["TypeOfDocument"] = df["TypeOfDocument"].astype('category').cat.codes

df['ror_1_shift1'] = df.groupby('SecuritiesCode')['ror_1'].shift(1) * -1
df['ror_1_shift2'] = df.groupby('SecuritiesCode')['ror_1'].shift(2)
df['ror_1_shift3'] = df.groupby('SecuritiesCode')['ror_1'].shift(3) * -1
df['ror_1_shift4'] = df.groupby('SecuritiesCode')['ror_1'].shift(4)
df['ror_1_shift5'] = df.groupby('SecuritiesCode')['ror_1'].shift(5)
df['ror_1_shift6'] = df.groupby('SecuritiesCode')['ror_1'].shift(6) * -1
df['ror_1_shift7'] = df.groupby('SecuritiesCode')['ror_1'].shift(7) * -1
df['ror_1_shift8'] = df.groupby('SecuritiesCode')['ror_1'].shift(8)
df['ror_1_shift9'] = df.groupby('SecuritiesCode')['ror_1'].shift(9)

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

time_config = {'val_split_date': '2021-11-01',
               'test_split_date': '2022-01-01'}

train = df[(df.Date < time_config['val_split_date'])]
val = df[(df.Date >= time_config['val_split_date']) & (df.Date < time_config['test_split_date'])]
test = df[(df.Date >= time_config['test_split_date'])]

print(train.shape)
print(val.shape)
print(test.shape)

train_groups = train.groupby('Date')['SecuritiesCode'].nunique()
query_train = train_groups.values
query_val = [val.shape[0] / 2000] * 2000
query_test = [test.shape[0] / 2000] * 2000

not_use_cols = ["RowId", "Date", "Target", "high_rank", "low_rank", "qcut", "DateCode", "DisclosedDate", "DisclosedTime", "CurrentPeriodEndDate", "TypeOfCurrentPeriod", "CurrentFiscalYearStartDate", "CurrentFiscalYearEndDate"]
col_use = [c for c in df.columns if c not in not_use_cols]
# col_use = ['Volume', '33SectorCode', '17SectorCode', 'Universe0', 'AdjustedOpen', 'AdjustedHigh', 'AdjustedLow', 'AdjustedClose',
#            'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_vortex_ind_pos',
#             'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff',
#             'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg',
#             'trend_cci', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
#             'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi', 'momentum_stoch_rsi_k',
#             'momentum_stoch_rsi_d', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_roc',
#             'momentum_ppo', 'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal', 'momentum_pvo_hist', 'momentum_kama',
#             'ror_1', 'ror_5', 'ror_10', 'ror_20', 'ror_40', 'ror_60', 'ror_100', 'TradedAmount_1', 'TradedAmount_5', 'TradedAmount_10',
#             'TradedAmount_20', 'TradedAmount_40', 'TradedAmount_60', 'TradedAmount_100', 'd_Amount', 'PreviousClose', 'range_1', 'range_5',
#             'range_10', 'range_20', 'range_40', 'range_60', 'range_100', 'd_atr', 'gap_range', 'gap_range_1', 'gap_range_5', 'gap_range_10', 'gap_range_20',
#             'gap_range_40', 'gap_range_60', 'gap_range_100', 'day_range_1', 'day_range_5', 'day_range_10', 'day_range_20', 'day_range_40',
#             'day_range_60', 'day_range_100', 'hig_range_1', 'hig_range_5', 'hig_range_10', 'hig_range_20', 'hig_range_40', 'hig_range_60',
#             'hig_range_100', 'mi_1', 'mi_5', 'mi_10', 'mi_20', 'mi_40', 'mi_60', 'mi_100', 'vola_5', 'vola_10', 'vola_20', 'vola_40', 'vola_60',
#             'vola_100', 'hl_5', 'hl_10', 'hl_20', 'hl_40', 'hl_60', 'hl_100', 'NetSales', 'OperatingProfit', 'OrdinaryProfit', 'Profit', 'EarningsPerShare',
#             'TotalAssets', 'Equity', 'EquityToAssetRatio', 'BookValuePerShare', 'ResultDividendPerShare1stQuarter', 'ResultDividendPerShare2ndQuarter',
#             'ResultDividendPerShare3rdQuarter', 'ResultDividendPerShareFiscalYearEnd', 'ResultDividendPerShareAnnual', 'NetSalesRatio', 'SecuritiesCode',
#             'DisclosedDate_diff', 'MarketCapitalization', 'MarketCapitalization/NetSales']
col_use  = [
    'd_atr',
    'day_range_100', 'vola_5', 'vola_10', 'vola_20', 'vola_40', 'vola_60', 'vola_100',
    'ror_1_shift1', 'ror_1_shift2', 'ror_1_shift3', 'ror_1_shift4', 'ror_1_shift5',
    'ror_1_shift6', 'ror_1_shift7', 'ror_1_shift8', 'ror_1_shift9']


model = LGBMRanker(
    # boosting_type="dart",
    # objective="lambdarank",
    metric="None",
    # n_estimators=100000,
    random_state=42,
    num_leaves=5,
    learning_rate=0.1,
    # subsample=0.9,
    # subsample_freq=5,
    # max_bin=20,
    # subsample_for_bin=20000,
    # colsample_bytree=0.9,
    n_jobs=cpu_count(),
    # reg_alpha=0.2,
    # reg_lambda=0.2,
    label_gain=np.arange(rank_num),
    )

model = LGBMRanker(
    metric="None",
    random_state=42,
    num_leaves=5,
    learning_rate=0.1,
    n_jobs=cpu_count(),
    label_gain=np.arange(rank_num),
    lambdarank_truncation_level=rank_num,
    max_bin=128,
    )

# max_labels = int(rank_num * 2)
# model = LGBMRanker(
#     # device="gpu",
#     boosting_type="dart",
#     objective="lambdarank",
#     metric="None",
#     label_gain=np.arange(max_labels),
#     lambdarank_truncation_level=max_labels,
#     num_iterations=1000//2,
#     # early_stopping_round=10,
#     num_leaves=2**4-1,
#     learning_rate=0.01,
#     max_bin=128,
#     # max_drop=0,
#     # bagging_freq=1,
#     # bagging_fraction=0.8,
#     # feature_fraction=0.5,
#     # lambdarank_norm=False,
#     # seed=123,
#     # min_data_in_leaf=100,
#     # min_sum_hessian_in_leaf=1e-2,
#     n_jobs=4,
# )

import ipdb; ipdb.set_trace()
model.fit(
    train[col_use], train['qcut'],
    group=train_groups,
    verbose=1,
    eval_set=[(val[col_use], val['qcut'])],
    eval_group=[query_val],
    # eval_at=[1],  # Make evaluation for target=1 ranking, I choosed arbitrarily
    eval_metric=custom_metric,
    early_stopping_rounds=100,
)

# # LGBMRegressor
# pred_model = LGBMRegressor(
#     n_estimators=15000,
#     random_state=42,
#     num_leaves=7,
#     learning_rate=0.002,
#     # subsample=0.8,
#     # subsample_freq=5,
#     # max_bin=20,
#     # subsample_for_bin=20000,
#     colsample_bytree=0.7,
#     n_jobs=cpu_count(),
# )
# # train
# pred_model.fit(
#     train[col_use], train['Target'],
#     verbose=1,
#     early_stopping_rounds=100,
#     eval_set=[(val[col_use], val['Target'])],
# )

# from sklearn.ensemble import RandomForestRegressor

# RFR = RandomForestRegressor()

test['predict'] = model.predict(test[col_use])
test = test.reset_index(drop=True)
test = test.sort_values(["Date", "predict"], ascending=[True, False])
ranking = test.groupby("Date").apply(set_rank).reset_index(drop=True)

# calc spread return sharpe
sharp_ratio, buf = calc_spread_return_sharpe(ranking, portfolio_size=200)
