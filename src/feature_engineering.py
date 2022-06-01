from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
# from __future__ import division
import ta

from utils import timer
import warnings
warnings.filterwarnings('ignore')


def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if (c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f"Mem. usage decreased to {end_mem:5.2f} MB" + \
        f" ({reduction * 100:.1f} % reduction)"
    if verbose:
        print(msg)

    return df


def get_time_info(df, is_onehot=False):
    datetime_series = pd.to_datetime(df['Date']).dt

    def get_day(dt_series):
        return dt_series.day

    def get_weekday(dt_series, is_onehot=False):
        if is_onehot:
            return pd.get_dummies(dt_series.day_name())
        return dt_series.dayofweek

    def get_week(dt_series):
        return dt_series.isocalendar().week

    def get_month(dt_series, is_onehot=False):
        if is_onehot:
            return pd.get_dummies(dt_series.month_name())
        return dt_series.month

    day = get_day(datetime_series)
    weekday = get_weekday(datetime_series, is_onehot)
    week = get_week(datetime_series)
    month = get_month(datetime_series, is_onehot)

    if is_onehot:
        return pd.concat([df, day, weekday, week, month], axis=1)
    df['day'] = day
    df['weekday'] = weekday
    df['week'] = week
    df['month'] = month
    return df


def both_fillna(df):
    ffill_df = df.groupby('SecuritiesCode').fillna(method='ffill')
    ffill_df['SecuritiesCode'] = df['SecuritiesCode']
    both_fill_df = ffill_df.groupby('SecuritiesCode').fillna(method='bfill')
    both_fill_df['SecuritiesCode'] = df['SecuritiesCode']
    return both_fill_df


def load_stock(is_all=True):
    val_date = ''
    if is_all:
        train_df = pd.read_csv("../Input/train_files/stock_prices.csv")
        val_df = pd.read_csv("../Input/supplemental_files/stock_prices.csv")
        val_date = val_df.iloc[0]['Date']
        df = pd.concat([train_df, val_df]).reset_index(drop=True)
    else:
        df = pd.read_csv("../Input/train_files/stock_prices.csv")
    stock_list = pd.read_csv("../Input/stock_list.csv")
    stock_list = stock_list[['SecuritiesCode', 'Section/Products', 'NewMarketSegment', '33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode', 'Universe0']]
    stock_list.replace('-', np.nan, inplace=True)

    return df, stock_list, val_date

# def load_stock(target_data):
#     if target_data == 'all':
#         train_df = pd.read_csv("../Input/train_files/stock_prices.csv")
#         val_df = pd.read_csv("../Input/supplemental_files/stock_prices.csv")
#         val_date = val_df.iloc[0]['Date']
#         df = pd.concat([train_df, val_df]).reset_index(drop=True)
#     else:
#         df = pd.read_csv("../Input/train_files/stock_prices.csv")
#     stock_list = pd.read_csv("../Input/stock_list.csv")
#     stock_list = stock_list[['SecuritiesCode', 'Section/Products', 'NewMarketSegment', '33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode', 'Universe0']]
#     stock_list.replace('-', np.nan, inplace=True)

#     return df, stock_list,


def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_price(df, price_cols=['Open', 'High', 'Low', 'Close']):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()

        for price_col in price_cols:
            # generate AdjustedClose
            df.loc[:, f"Adjusted{price_col}"] = (
                df["CumulativeAdjustmentFactor"] * df[price_col]
            ).map(lambda x: float(
                Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
            ))
            # reverse order
            df = df.sort_values("Date")
            # to fill AdjustedClose, replace 0 into np.nan
            df.loc[df[f"Adjusted{price_col}"] == 0, f"Adjusted{price_col}"] = np.nan
            # forward fill AdjustedClose
            df.loc[:, f"Adjusted{price_col}"] = df.loc[:, f"Adjusted{price_col}"].ffill()
        # return df["Date",  f"Adjusted{price_col}"]
        return df

    # generate Adjusted Prices
    price = price.sort_values(["Date", "SecuritiesCode"])
    AdjustedPrices = price.groupby("SecuritiesCode").apply(generate_adjusted_price).reset_index(drop=True)

    return AdjustedPrices


def shift_period_pearson_corr(df, target_col, index, columns, period):
    # 2000銘柄が揃うのは2020-12-23以降
    df = df[df['Date'] >= '2020-12-23']
    pivot = df.pivot(index=index, columns=columns, values=target_col)
    pivot.fillna(method='ffill', inplace=True)
    pivot.fillna(method='bfill', inplace=True)

    target = pivot.iloc[period:, :].values
    # target_lag = pivot.shift(period).iloc[period:, :].values
    target_lag = pivot.iloc[:-period, :].values

    diff = target - target.mean()
    lag_diff = target_lag - target_lag.mean()

    lag_corr = np.dot(diff.T, lag_diff) / (np.sqrt(sum(diff ** 2)) * np.sqrt(sum(lag_diff ** 2)))
    for i in range(lag_corr.shape[0]):
        lag_corr[i, i] = 0.0

    return lag_corr


def append_lag_corr_code(df, target_col='AdjustedClose', index='Date', columns='SecuritiesCode', period=1):
    new_df = df.copy()

    lag_corr = shift_period_pearson_corr(new_df, target_col, index, columns, period)

    lag_idxmax = lag_corr.argmax(axis=1)
    lag_idxmin = lag_corr.argmin(axis=1)

    lag_corr_master = pd.DataFrame()
    lag_corr_master['SecuritiesCode'] = sorted(new_df['SecuritiesCode'].unique())
    lag_corr_master = lag_corr_master.reset_index()

    lag_idxmax_dict = dict(zip(range(lag_corr_master.shape[0]), lag_idxmax))
    lag_idxmin_dict = dict(zip(range(lag_corr_master.shape[0]), lag_idxmin))
    code_master_dict = dict(zip(range(lag_corr_master.shape[0]),  sorted(new_df['SecuritiesCode'].unique())))
    lag_corr_master['lag_max_corr_code'] = lag_corr_master['index'].map(lag_idxmax_dict)
    lag_corr_master['lag_min_corr_code'] = lag_corr_master['index'].map(lag_idxmin_dict)
    lag_corr_master[f'{target_col}_lag{period}_max_corr_code'] = lag_corr_master['lag_max_corr_code'].map(code_master_dict)
    lag_corr_master[f'{target_col}_lag{period}_min_corr_code'] = lag_corr_master['lag_min_corr_code'].map(code_master_dict)
    lag_corr_master = lag_corr_master[['SecuritiesCode', f'{target_col}_lag{period}_max_corr_code', f'{target_col}_lag{period}_min_corr_code']]

    return new_df.merge(lag_corr_master, on="SecuritiesCode", how="left")


def append_value_by_corr(df, target_col='', target_corr_cols=[], suffixes=('', '_corr'), is_rowid=False):
    new_df = df.copy()

    if is_rowid:
        target_dict = dict(zip(new_df['RowId'], new_df[target_col]))
        if isinstance(target_corr_cols, list):
            for col in target_corr_cols:
                new_df[f'{col}_RowId'] = new_df['Date'].str.replace(r'-', '') + '_' + new_df[col].astype(str)
                target_corr_col = col[:-5]
                new_df[target_corr_col] = new_df[f'{col}_RowId'].map(target_dict)
                new_df.drop(columns=[f'{col}_RowId'], inplace=True)
        return new_df
    else:
        dfs = []
        for corr_col in target_corr_cols:
            merge_df = new_df.merge(
                    new_df[['Date', 'SecuritiesCode', target_col]],
                    left_on=['Date', corr_col],
                    right_on=['Date', 'SecuritiesCode'],
                    suffixes=('', '_corr'),
                    how='left'
                ).iloc[:, -1].to_frame()
            merge_df.columns = [f'{target_col}_{corr_col[:-10]}']
            # new_df.drop(columns=[corr_col], inplace=True)
            dfs.append(
                merge_df
            )

        return pd.concat(dfs, axis=1)


def unpivot_price(df):
    unpivot_df = pd.melt(
        df[['Date', 'SecuritiesCode', 'AdjustedOpen', 'AdjustedHigh', 'AdjustedLow', 'AdjustedClose']],
        id_vars=['Date', 'SecuritiesCode'],
        value_vars=['AdjustedOpen',	'AdjustedHigh', 'AdjustedLow', 'AdjustedClose']
    )
    return unpivot_df.sort_values(['Date', 'SecuritiesCode']).reset_index(drop=True)


def calc_unpivot_target(df, method='original'):
    unpivot_df = unpivot_price(df)
    unpivot_df = unpivot_df.merge(
        df[['Date', 'SecuritiesCode', 'Target']],
        on=['Date', 'SecuritiesCode'],
        how='left'
    )
    if method == 'original':
        return unpivot_df

    # unpivot_group = unpivot_df.groupby(['SecuritiesCode', 'variable'])
    # unpivot_df['Target'] = (unpivot_group['value'].diff() / unpivot_group.shift(1)['value'])
    unpivot_df['Target'] = unpivot_df.groupby(['SecuritiesCode', 'variable'])['value'].pct_change(1)
    unpivot_df['Target'].fillna(0, inplace=True)
    return unpivot_df


def aggregate_window(df, group_col='SecuritiesCode', target_col='AdjustedClose', window=5):
    '''
    Aggregate target columns by window as feature engineering.
    '''
    agg_columns = ['mean', 'var', 'max', 'min', 'skew', 'kurt']
    agg_df = (
        df.groupby(group_col)[target_col]
            .rolling(window)
            .agg(agg_columns)
            .reset_index()
            .set_index('level_1')
            .sort_values('level_1')
    )
    agg_df.drop(columns=['SecuritiesCode'], inplace=True)
    agg_df.columns = [f'{target_col}_{col}_{window}' for col in agg_columns]

    ewm_agg_columns = ['mean', 'std']
    ewm_agg_df = (
        df.groupby(group_col)[target_col]
            .ewm(com=0.5)
            .agg(ewm_agg_columns)
            .reset_index()
            .set_index('level_1')
            .sort_values('level_1')
    )
    ewm_agg_df.drop(columns=['SecuritiesCode'], inplace=True)
    ewm_agg_df.columns = [f'{target_col}_ewm_{col}_{window}' for col in ewm_agg_columns]

    return pd.concat([df, agg_df, ewm_agg_df], axis=1)


def datediff(df, date='Date', disclosed_date='DisclosedDate'):
    df[[date, disclosed_date]] = df[[date, disclosed_date]].apply(pd.to_datetime) #if conversion required
    df[f'{disclosed_date}_diff'] = (df[date] - df[disclosed_date]).dt.days
    return df


def change_ratio(df):
    # 騰落率
    df["ror_1"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(1)
    df["ror_2"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(2)
    df["ror_3"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(3)
    df["ror_4"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(4)
    df["ror_5"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(5)
    df["ror_6"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(6)
    df["ror_7"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(7)
    df["ror_8"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(8)
    df["ror_9"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(9)
    df["ror_10"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(10)
    df["ror_20"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(20)
    df["ror_40"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(40)
    df["ror_60"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(60)
    df["ror_100"] = df.groupby('SecuritiesCode')['AdjustedClose'].pct_change(100)
    return df


def rolling_mean(df, target_col):
    groups = df.groupby('SecuritiesCode')
    rolling_df = pd.DataFrame()
    df[f'{target_col}_1'] = df[target_col]
    rolling_df[['level_1', f'{target_col}_5']] = groups[target_col].rolling(5).mean().reset_index()[['level_1', target_col]]
    rolling_df[f'{target_col}_10'] = groups[target_col].rolling(10).mean().reset_index(drop=True)
    rolling_df[f'{target_col}_20'] = groups[target_col].rolling(20).mean().reset_index(drop=True)
    rolling_df[f'{target_col}_40'] = groups[target_col].rolling(40).mean().reset_index(drop=True)
    rolling_df[f'{target_col}_60'] = groups[target_col].rolling(60).mean().reset_index(drop=True)
    rolling_df[f'{target_col}_100'] = groups[target_col].rolling(100).mean().reset_index(drop=True)
    rolling_df = rolling_df.set_index('level_1').sort_values('level_1')

    return pd.concat([df, rolling_df], axis=1)


def rolling_methods(df):
    df = change_ratio(df)

    # 売買代金
    df["TradedAmount"] = df["AdjustedClose"] * df["Volume"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = rolling_mean(df, "TradedAmount")
    df["d_Amount"] = df["TradedAmount"] / df["TradedAmount_20"]

    # レンジ
    df['PreviousClose'] = df.groupby('SecuritiesCode')['AdjustedClose'].shift(1)
    df["range"] = (df[['PreviousClose', 'AdjustedHigh']].max(axis=1) - df[['PreviousClose', 'AdjustedLow']].min(axis=1)) / df['PreviousClose']
    df = df.replace([np.inf, -np.inf], np.nan)
    df = rolling_mean(df, "range")
    df["d_atr"] = df["range"] / df["range_20"]

    # ギャップレンジ
    df["gap_range"] = (np.abs(df["AdjustedOpen"] - df["PreviousClose"])) / df["PreviousClose"]
    df = rolling_mean(df, "gap_range")

    # デイレンジ
    df["day_range"] = (df["AdjustedHigh"] - df["AdjustedLow"]) / df["PreviousClose"]
    df = rolling_mean(df, "day_range")

    # ヒゲレンジ
    df["hig_range"] = ((df["AdjustedHigh"] - df["AdjustedLow"]) - np.abs(df["AdjustedOpen"] - df["AdjustedClose"])) / df["PreviousClose"]
    df = rolling_mean(df, "hig_range")

    # マーケットインパクト
    df["mi"] = df["range"] / (df["Volume"] * df["AdjustedClose"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = rolling_mean(df, "mi")

    # ボラティリティ
    groups = df.groupby('SecuritiesCode')
    rolling_df = pd.DataFrame()
    rolling_df[['level_1', "vola_5"]] = groups["ror_1"].rolling(5).std().reset_index()[['level_1', 'ror_1']]
    rolling_df["vola_10"] = groups["ror_1"].rolling(10).std().reset_index(drop=True)
    rolling_df["vola_20"] = groups["ror_1"].rolling(20).std().reset_index(drop=True)
    rolling_df["vola_40"] = groups["ror_1"].rolling(40).std().reset_index(drop=True)
    rolling_df["vola_60"] = groups["ror_1"].rolling(60).std().reset_index(drop=True)
    rolling_df["vola_100"] = groups["ror_1"].rolling(100).std().reset_index(drop=True)
    df = pd.concat([df, rolling_df], axis=1)

    # HLバンド
    groups = df.groupby('SecuritiesCode')
    rolling_df = pd.DataFrame()
    rolling_df[["level_1", "hl_5"]] = (groups["AdjustedHigh"].rolling(5).max() - groups["AdjustedLow"].rolling(5).min()).reset_index()[["level_1",  0]]
    rolling_df["hl_10"] = (groups["AdjustedHigh"].rolling(10).max() - groups["AdjustedLow"].rolling(10).min()).reset_index(drop=True)
    rolling_df["hl_20"] = (groups["AdjustedHigh"].rolling(20).max() - groups["AdjustedLow"].rolling(20).min()).reset_index(drop=True)
    rolling_df["hl_40"] = (groups["AdjustedHigh"].rolling(40).max() - groups["AdjustedLow"].rolling(40).min()).reset_index(drop=True)
    rolling_df["hl_60"] = (groups["AdjustedHigh"].rolling(60).max() - groups["AdjustedLow"].rolling(60).min()).reset_index(drop=True)
    rolling_df["hl_100"] = (groups["AdjustedHigh"].rolling(100).max() - groups["AdjustedLow"].rolling(100).min()).reset_index(drop=True)
    df = pd.concat([df, rolling_df], axis=1)

    return df


def aggregate(df, val_date):
    # if target_data == 'all':
    #     train_df = pd.read_csv("../Input/train_files/stock_prices.csv")
    #     val_df = pd.read_csv("../Input/supplemental_files/stock_prices.csv")
    #     val_date = val_df.iloc[0]['Date']
    #     df = pd.concat([train_df, val_df]).reset_index(drop=True)
    # else:
    #     df = pd.read_csv("../Input/train_files/stock_prices.csv")
    # stock_list = pd.read_csv("../Input/stock_list.csv")
    # stock_list = stock_list[['SecuritiesCode', 'Section/Products', 'NewMarketSegment', '33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode', 'Universe0']]
    # stock_list.replace('-', np.nan, inplace=True)

    # df = df.merge(stock_list, on='SecuritiesCode', how='left')
    df = get_time_info(df)
    print(df.shape)
    print(df.isnull().sum())

    df['Target'].fillna(0, inplace=True)
    df['ExpectedDividend'].fillna(0, inplace=True)
    df = both_fillna(df)
    df["high_rank"] = df.groupby("Date")["Target"].rank("dense", ascending=False).astype(int)

    print(df.isnull().sum())

    df = adjust_price(df)
    df = rolling_methods(df)

    df['ror_1_shift1'] = df.groupby('SecuritiesCode')['ror_1'].shift(1)
    df['ror_1_shift2'] = df.groupby('SecuritiesCode')['ror_1'].shift(2)
    df['ror_1_shift3'] = df.groupby('SecuritiesCode')['ror_1'].shift(3)
    df['ror_1_shift4'] = df.groupby('SecuritiesCode')['ror_1'].shift(4)
    df['ror_1_shift5'] = df.groupby('SecuritiesCode')['ror_1'].shift(5)
    df['ror_1_shift6'] = df.groupby('SecuritiesCode')['ror_1'].shift(6)
    df['ror_1_shift7'] = df.groupby('SecuritiesCode')['ror_1'].shift(7)
    df['ror_1_shift8'] = df.groupby('SecuritiesCode')['ror_1'].shift(8)
    df['ror_1_shift9'] = df.groupby('SecuritiesCode')['ror_1'].shift(9)

    with timer('ta.add_all_ta_features'):
        all_ta_features = (
            df[['SecuritiesCode', "AdjustedOpen", "AdjustedHigh", "AdjustedLow", "AdjustedClose", "Volume"]]
            .groupby('SecuritiesCode')
            .apply(
                    lambda x: ta.add_all_ta_features(
                        x, "AdjustedOpen", "AdjustedHigh", "AdjustedLow", "AdjustedClose", "Volume", fillna=False
                    )
                )
        )
    all_ta_features.drop(columns=["AdjustedOpen", "AdjustedHigh", "AdjustedLow", "AdjustedClose", "Volume"], inplace=True)
    df = pd.concat([df, all_ta_features.iloc[:, 1:]], axis=1)

    df.memory_usage(index=True)

    Open_lag1_corr = append_lag_corr_code(df, target_col='AdjustedOpen')
    Close_lag1_corr = append_lag_corr_code(df, target_col='AdjustedClose')
    Target_lag1_corr = append_lag_corr_code(df, target_col='Target')
    high_rank_lag1_corr = append_lag_corr_code(df, target_col='high_rank')

    ror_1_corr = append_value_by_corr(Open_lag1_corr, target_col='ror_1', target_corr_cols=['AdjustedOpen_lag1_max_corr_code', 'AdjustedOpen_lag1_min_corr_code'])
    d_Amount_corr = append_value_by_corr(Close_lag1_corr, target_col='d_Amount', target_corr_cols=['AdjustedClose_lag1_max_corr_code', 'AdjustedClose_lag1_min_corr_code'])
    TradedAmount_1_corr = append_value_by_corr(Close_lag1_corr, target_col='TradedAmount_1', target_corr_cols=['AdjustedClose_lag1_max_corr_code', 'AdjustedClose_lag1_min_corr_code'])
    d_atr_corr = append_value_by_corr(Target_lag1_corr, target_col='d_atr', target_corr_cols=['Target_lag1_max_corr_code', 'Target_lag1_min_corr_code'])
    day_range_corr = append_value_by_corr(Target_lag1_corr, target_col='day_range_1', target_corr_cols=['Target_lag1_max_corr_code', 'Target_lag1_min_corr_code'])
    vola_5_corr = append_value_by_corr(high_rank_lag1_corr, target_col='vola_5', target_corr_cols=['high_rank_lag1_max_corr_code', 'high_rank_lag1_min_corr_code'])
    gap_range_1_corr = append_value_by_corr(high_rank_lag1_corr, target_col='gap_range_1', target_corr_cols=['high_rank_lag1_max_corr_code', 'high_rank_lag1_min_corr_code'])

    concat_df = pd.concat([df, ror_1_corr, d_Amount_corr, TradedAmount_1_corr, d_atr_corr, day_range_corr, vola_5_corr, gap_range_1_corr], axis=1)

    agg_df = aggregate_window(concat_df, target_col='ror_1')
    agg_df.drop(columns=('level_1'), inplace=True)
    if val_date:
        train_agg_df = agg_df[agg_df['Date'] < val_date]
        val_agg_df = agg_df[agg_df['Date'] >= val_date]
        train_agg_df.to_parquet('../Output/train_agg_df.parquet')
        val_agg_df.to_parquet('../Output/val_agg_df.parquet')
    else:
        agg_df.to_parquet('../Output/train_agg_df.parquet')

    return agg_df


def preprocess_financials(agg_df, val_date):
    if val_date:
        train_fin = pd.read_csv('../Input/train_files/financials.csv')
        val_fin = pd.read_csv('../Input/supplemental_files/financials.csv')
        financials = pd.concat([train_fin, val_fin]).reset_index(drop=True)
    else:
        financials = pd.read_csv('../Input/train_files/financials.csv')

    numeric_cols = [
        'NetSales', 'OperatingProfit',
        'OrdinaryProfit', 'Profit', 'EarningsPerShare', 'TotalAssets', 'Equity',
        'EquityToAssetRatio', 'BookValuePerShare',
        'ResultDividendPerShare1stQuarter',
        'ResultDividendPerShare2ndQuarter',
        'ResultDividendPerShare3rdQuarter',
        'ResultDividendPerShareFiscalYearEnd',
        'ResultDividendPerShareAnnual',
        'ForecastDividendPerShare1stQuarter',
        'ForecastDividendPerShare2ndQuarter',
        'ForecastDividendPerShare3rdQuarter',
        'ForecastDividendPerShareFiscalYearEnd',
        'ForecastDividendPerShareAnnual',
        'ForecastNetSales',
        'ForecastOperatingProfit',
        'ForecastOrdinaryProfit',
        'ForecastProfit',
        'ForecastEarningsPerShare',
        'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock',
        'NumberOfTreasuryStockAtTheEndOfFiscalYear',
        'AverageNumberOfShares'
    ]
    str_cols = [col for col in financials.columns if col not in numeric_cols]

    fin_numeric = pd.to_numeric(financials[numeric_cols].stack(), errors='coerce').unstack()
    fin_str = financials[str_cols]
    financials = pd.concat([fin_str, fin_numeric], axis=1)

    financials_group = financials.groupby(['SecuritiesCode'])
    financials['NetSalesRatio'] = (financials_group['NetSales'].diff() / financials_group.shift(1)['NetSales'])

    # drop_cols = [col for col in agg_df.columns if col not in ['RowId', 'Date', 'SecuritiesCode']]
    fin_df = financials.drop(columns=['Date', 'SecuritiesCode'])
    fin_agg_df = agg_df.merge(fin_df, left_on='RowId', right_on='DateCode', how='left')
    fin_agg_df.drop_duplicates(subset='RowId', keep='last', inplace=True)
    fill_fin_df = both_fillna(fin_agg_df)
    fill_fin_df = datediff(fill_fin_df)

    fill_fin_df['MarketCapitalization'] = fill_fin_df['AdjustedClose'] * fill_fin_df['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']
    fill_fin_df['MarketCapitalization/NetSales'] = fill_fin_df['MarketCapitalization'] / fill_fin_df['NetSales']
    # fill_fin_df.drop(columns=('level_1'), inplace=True)

    fill_fin_df["Section/Products"] = fill_fin_df["Section/Products"].astype('category').cat.codes
    fill_fin_df["NewMarketSegment"] = fill_fin_df["NewMarketSegment"].astype('category').cat.codes
    fill_fin_df["TypeOfDocument"] = fill_fin_df["TypeOfDocument"].astype('category').cat.codes
    fill_fin_df.loc[:, ['33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode', 'week']] = \
        pd.to_numeric(fill_fin_df[['33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode', 'week']].stack(), errors='coerce').unstack()

    if val_date:
        financials_train_agg_df = fill_fin_df[fill_fin_df['Date'] < val_date]
        financials_val_agg_df = fill_fin_df[fill_fin_df['Date'] >= val_date]
        financials_train_agg_df.to_parquet('../Output/financials_train_agg_df.parquet')
        financials_val_agg_df.to_parquet('../Output/financials_val_agg_df.parquet')
    else:
        fill_fin_df.to_parquet('../Output/financials_train_agg_df.parquet')

    return fill_fin_df


def unpivot(fin_df, val_date):
    unpivot_df = calc_unpivot_target(fin_df, method='calc')
    unpivot_fin_df = pd.merge(
        unpivot_df,
        fin_df,
        on=['Date', 'SecuritiesCode'],
        how='left',
        suffixes=('_unpivot', '')
    )
    if val_date:
        unpivot_train_df = unpivot_fin_df[unpivot_fin_df['Date'] < val_date]
        unpivot_val_df = unpivot_fin_df[unpivot_fin_df['Date'] >= val_date]
        unpivot_train_df.to_parquet('../Output/unpivot_train_df.parquet')
        unpivot_val_df.to_parquet('../Output/unpivot_val_df.parquet')
    else:
        unpivot_fin_df.to_parquet('../Output/unpivot_train_df.parquet')


def run(is_all=True):
    df, stock_list, val_date = load_stock(is_all)
    df = df.merge(stock_list, on='SecuritiesCode', how='left')
    with timer('aggregate'):
        agg_df = aggregate(df, val_date)
    with timer('preprocess_financials'):
        fill_fin_df = preprocess_financials(agg_df, val_date)
    with timer('unpivot'):
        unpivot(fill_fin_df, val_date)


if __name__ == '__main__':
    with timer('feature_engineering'):
        run()
