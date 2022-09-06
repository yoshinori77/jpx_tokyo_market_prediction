import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from lightgbm import LGBMRanker, LGBMRegressor
from multiprocessing import cpu_count
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

import lightgbm as lgb
import time

from contextlib import contextmanager


@contextmanager
def timer(name: str):
    t0 = time.time()
    msg = f"[{name}] start"
    print(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    print(msg)



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


# self defined GroupTimeSeriesSplit
class GroupTimeSeriesSplit(_BaseKFold):

    def __init__(self, n_splits=5, *, max_train_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        n_splits = self.n_splits
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        group_counts = np.unique(groups, return_counts=True)[1]
        groups = np.split(indices, np.cumsum(group_counts)[:-1])
        n_groups = _num_samples(groups)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of groups: {1}.").format(n_folds, n_groups))
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        for test_start in test_starts:
            if self.max_train_size:
                train_start = np.searchsorted(
                    np.cumsum(
                        group_counts[:test_start][::-1])[::-1] < self.max_train_size + 1, 
                        True)
                yield (np.concatenate(groups[train_start:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))
            else:
                yield (np.concatenate(groups[:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))


def lgbm_model_rank(train_x, train_y, validation_x, validation_y, query_train, query_validation, rank_num=100):
    # params = {
    #     # baseline parameters
    #     "objective" : "lambdarank",
    #     # "objective" : "rank_xendcg",
    #     # "metric" : "None",
    #     "num_leaves" : 23,
    #     "learning_rate" : 0.01,
    #     # "bagging_fraction" : 0.6,
    #     # "feature_fraction" : 0.6,
    #     "bagging_seed" : 42,
    #     "verbosity" : -1,
    #     "seed": 42,
    #     # "num_class": 6,
    #     # "max_bin": 128,
    #     # "colsample_bytree": 0.9,
    #     # n_jobs=cpu_count(),
    #     # "reg_alpha": 0.2,
    #     # "reg_lambda": 0.2,
    #     "label_gain": np.arange(rank_num),
    #     "lambdarank_truncation_level": rank_num,
    #     # "ndcg_eval_at": np.concatenate([np.arange(200), np.arange(query_train.min()-200, query_train.min())]),
    #     "ndcg_eval_at": [1]
    # }
    params = {
        # baseline parameters
        "objective": "lambdarank",
        # "metric": "map",
        "num_leaves": 11,
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_seed": 42,
        "verbosity": -1,
        "seed": 42,
        # "num_class": 6,
        "max_bin": 128,
        # "colsample_bytree": 0.9,
        # n_jobs=cpu_count(),
        # "reg_alpha": 0.2,
        # "reg_lambda": 0.2,
        "label_gain": np.arange(rank_num),
        # "lambdarank_truncation_level": rank_num,
        # "ndcg_eval_at": np.concatenate([np.arange(200), np.arange(query_train.min()-200, query_train.min())]),
        "eval_at": [1]
    }

    lg_train = lgb.Dataset(train_x, label=train_y, group=query_train)
    lg_validation = lgb.Dataset(validation_x, label=validation_y, group=query_validation)
    evals_result_lgbm = {}
    
    model_lightgbm = lgb.train(params, lg_train, valid_sets=[lg_validation], num_boost_round=2000,
                               early_stopping_rounds=50, evals_result=evals_result_lgbm, verbose_eval=5,
                            #    eval_at=[1],  # Make evaluation for target=1 ranking, I choosed arbitrarily
                            #    feval=custom_metric,
                               )
    
    # model_lightgbm.save_model(f'model_lightgbm_{index}.txt')
    # pre_test_lightgbm = model_lightgbm.predict(test_x, num_iteration=model_lightgbm.best_iteration)

    # return pre_test_lightgbm, model_lightgbm, evals_result_lgbm
    return model_lightgbm, evals_result_lgbm


with timer('load data'):
    unpivot_train = pd.read_parquet('../Output/unpivot_train_df.parquet')
    unpivot_val = pd.read_parquet('../Output/unpivot_val_df.parquet')

with timer('concat data'):
    unpivot_df = pd.concat([unpivot_train, unpivot_val]).reset_index(drop=True)
    rank_num = 100
    unpivot_df["qcut"] = pd.qcut(unpivot_df['Target'], rank_num, duplicates='drop').cat.codes

del unpivot_train, unpivot_val

with timer('cat codes'):
    unpivot_df["Section/Products"] = unpivot_df["Section/Products"].astype('category').cat.codes
    unpivot_df["NewMarketSegment"] = unpivot_df["NewMarketSegment"].astype('category').cat.codes
    unpivot_df["33SectorCode"] = unpivot_df["33SectorCode"].astype('category').cat.codes
    unpivot_df["17SectorCode"] = unpivot_df["17SectorCode"].astype('category').cat.codes
    unpivot_df["NewIndexSeriesSizeCode"] = unpivot_df["NewIndexSeriesSizeCode"].astype('category').cat.codes
    unpivot_df["week"] = unpivot_df["week"].astype('category').cat.codes
    unpivot_df["TypeOfDocument"] = unpivot_df["TypeOfDocument"].astype('category').cat.codes
    unpivot_df["variable"] = unpivot_df["variable"].astype('category').cat.codes

with timer('sort'):
    unpivot_df.sort_values(["Date", "variable", "SecuritiesCode"], ascending=[True, True, True], inplace=True)

unpivot_df = unpivot_df.replace([np.inf, -np.inf], np.nan).fillna(0)

with timer('split data'):
    time_config = {'val_split_date': '2021-11-01',
                   'test_split_date': '2022-01-01'}

    unpivot_train = unpivot_df[(unpivot_df.Date < time_config['val_split_date'])]
    unpivot_val = unpivot_df[(unpivot_df.Date >= time_config['val_split_date']) & (unpivot_df.Date < time_config['test_split_date'])]
    unpivot_test = unpivot_df[(unpivot_df.Date >= time_config['test_split_date'])]

del unpivot_df


col_use = [
    'variable', 'value',
    # 'Volume', 'NewMarketSegment', '33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode', 'Universe0',
    'day', 'weekday', 'week', 'month',
    'ror_1', 'ror_2', 'ror_3', 'ror_4', 'ror_5', 'ror_6', 'ror_7', 'ror_8', 'ror_9', 'ror_10', 'ror_20',
    # 'TradedAmount_1', 'TradedAmount_5', 'TradedAmount_10', 'd_Amount',
    # 'range_1', 'range_5', 'range_10', 'range_20', 'd_atr',
    # 'gap_range_1', 'gap_range_5', 'gap_range_10', 'gap_range_20',
    # 'day_range_1', 'day_range_5', 'day_range_10', 'day_range_20',
    # 'hig_range_1', 'hig_range_5', 'hig_range_10', 'hig_range_20',
    # 'mi_1', 'mi_5', 'mi_10', 'mi_20',
    # 'vola_5', 'vola_10', 'vola_20', 'vola_40',
    # 'hl_5', 'hl_10', 'hl_20', 'hl_40',
    'ror_1_shift1', 'ror_1_shift2', 'ror_1_shift3', 'ror_1_shift4', 'ror_1_shift5',
    'ror_1_shift6', 'ror_1_shift7', 'ror_1_shift8', 'ror_1_shift9',
    'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
    'momentum_rsi', 'momentum_wr',
    # 'ror_1_AdjustedOpen_lag1_max',
    # 'ror_1_AdjustedOpen_lag1_min',
    # 'd_Amount_AdjustedClose_lag1_max',
    # 'd_Amount_AdjustedClose_lag1_min',
    # 'TradedAmount_1_AdjustedClose_lag1_max',
    # 'TradedAmount_1_AdjustedClose_lag1_min',
    # 'd_atr_Target_lag1_max',
    # 'd_atr_Target_lag1_min',
    # 'day_range_1_Target_lag1_max',
    # 'day_range_1_Target_lag1_min',
    # 'vola_5_high_rank_lag1_max',
    # 'vola_5_high_rank_lag1_min',
    # 'gap_range_1_high_rank_lag1_max',
    # 'gap_range_1_high_rank_lag1_min',
    # 'ror_1_mean_5', 'ror_1_var_5', 'ror_1_max_5', 'ror_1_min_5',
    # 'ror_1_skew_5', 'ror_1_kurt_5', 'ror_1_ewm_mean_5', 'ror_1_ewm_std_5',
    'TypeOfDocument',
    # 'NetSales', 'Profit', 'Equity',
    # 'NetSalesRatio',
    'DisclosedDate_diff',
    # 'MarketCapitalization'
]

col_use = [
    'variable', 'value',
    # 'Volume', 'NewMarketSegment', '33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode', 'Universe0',
    'day', 'weekday', 'week', 'month',
    'ror_1', 'ror_2', 'ror_3', 'ror_4', 'ror_5', 'ror_6', 'ror_7', 'ror_8', 'ror_9', 'ror_10', 'ror_20',
    # 'TradedAmount_1', 'TradedAmount_5', 'TradedAmount_10', 'd_Amount',
    # 'range_1', 'range_5', 'range_10', 'range_20', 'd_atr',
    # 'gap_range_1', 'gap_range_5', 'gap_range_10', 'gap_range_20',
    # 'day_range_1', 'day_range_5', 'day_range_10', 'day_range_20',
    # 'hig_range_1', 'hig_range_5', 'hig_range_10', 'hig_range_20',
    # 'mi_1', 'mi_5', 'mi_10', 'mi_20',
    # 'vola_5', 'vola_10', 'vola_20', 'vola_40',
    # 'hl_5', 'hl_10', 'hl_20', 'hl_40',
    'ror_1_shift1', 'ror_1_shift2', 'ror_1_shift3', 'ror_1_shift4', 'ror_1_shift5',
    'ror_1_shift6', 'ror_1_shift7', 'ror_1_shift8', 'ror_1_shift9',
    'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
    'momentum_rsi', 'momentum_wr',
    'ror_1_AdjustedOpen_lag1_max',
    'ror_1_AdjustedOpen_lag1_min',
    'd_Amount_AdjustedClose_lag1_max',
    'd_Amount_AdjustedClose_lag1_min',
    'TradedAmount_1_AdjustedClose_lag1_max',
    'TradedAmount_1_AdjustedClose_lag1_min',
    'd_atr_Target_lag1_max',
    'd_atr_Target_lag1_min',
    'day_range_1_Target_lag1_max',
    'day_range_1_Target_lag1_min',
    'vola_5_high_rank_lag1_max',
    'vola_5_high_rank_lag1_min',
    'gap_range_1_high_rank_lag1_max',
    'gap_range_1_high_rank_lag1_min',
    'ror_1_mean_5', 'ror_1_var_5', 'ror_1_max_5', 'ror_1_min_5',
    'ror_1_skew_5', 'ror_1_kurt_5', 'ror_1_ewm_mean_5', 'ror_1_ewm_std_5',
    'TypeOfDocument',
    # 'NetSales', 'Profit', 'Equity',
    # 'NetSalesRatio',
    'DisclosedDate_diff',
    # 'MarketCapitalization'
]
unpivot_train_x = unpivot_train[col_use]
unpivot_train_y = unpivot_train[['qcut']]
# train_y = train[['high_rank']]
unpivot_groups = unpivot_train[['Date']]


sharp_ratio_list = []

import ipdb; ipdb.set_trace()

gtscv = GroupTimeSeriesSplit(n_splits=3, max_train_size=None)
for index, (train_id, val_id) in enumerate(gtscv.split(unpivot_train_x, groups=unpivot_groups)):
    train_groups = unpivot_train.iloc[train_id].groupby(['Date', 'variable'])['SecuritiesCode'].nunique().values
    val_groups = unpivot_train.iloc[val_id].groupby(['Date', 'variable'])['SecuritiesCode'].nunique().values

    model_lightgbm, evals_result_lgbm = lgbm_model_rank(unpivot_train_x.iloc[train_id], unpivot_train_y.iloc[train_id], unpivot_train_x.iloc[val_id], unpivot_train_y.iloc[val_id], train_groups, val_groups)
    unpivot_test['predict'] = model_lightgbm.predict(unpivot_test[col_use])
    unpivot_test = unpivot_test.sort_values(["Date", "predict"], ascending=[True, False])
    ranking = unpivot_test.groupby("Date").apply(set_rank).reset_index(drop=True)
    sharp_ratio, _ = calc_spread_return_sharpe(ranking, portfolio_size=200)
    sharp_ratio_list.append(sharp_ratio)
    print(sharp_ratio)
    ax = lgb.plot_metric(evals_result_lgbm)
    plt.show()
    print('Output of LightGBM Model training..')
