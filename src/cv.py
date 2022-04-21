import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.base import clone


def custom_metric(y, t):
    score = spearmanr(t, y, nan_policy="propagate")[0]
    return 'rho', score, True


def get_scores(t, y, full=True):
    r_high = spearmanr(t[:, 0], y[:, 0], nan_policy="propagate")[0]
    r_low = spearmanr(t[:, 1], y[:, 1], nan_policy="propagate")[0]
    score = (r_high - 1) ** 2 + (r_low - 1) ** 2
    if full:
        return r_high, r_low, score
    else:
        return score


def limit_range(X, y, year):
    from_date = pd.Timestamp(f"{year}-03-27")
    to_date = pd.Timestamp(f"{year}-05-15")
    dates = X.index.get_level_values("base_date")
    range_cnd = (dates >= from_date) & (dates <= to_date)
    # range_cnd &= X.index.get_level_values("prediction_target") == True

    X["high_20"] = y[:, 0]
    X["low_20"] = y[:, 1]
    X = X[range_cnd].groupby("Local Code").nth(-1)
    y = X[["high_20", "low_20"]].values
    X.drop(["high_20", "low_20"], axis=1, inplace=True)
    return X, y


def cv(estimator, X, y, w=None):
    estimators = []
    scores = []
    
    years = X.index.get_level_values("base_date").year
    uniq_years = np.array([2017, 2018, 2019, 2020, 2021])
    
    kf = KFold(n_splits=uniq_years.shape[0], shuffle=False)
    
    print(uniq_years)
    for tr_idx, ts_idx in kf.split(uniq_years):
        tr_years = uniq_years[tr_idx]
        ts_years = uniq_years[ts_idx]
        tr_mask = years.isin(tr_years)
        ts_mask = years.isin(ts_years)
        tr_X = X[tr_mask].copy()
        ts_X = X[ts_mask].copy()
        tr_y = y[tr_mask]
        ts_y = y[ts_mask]

        print(tr_years)
        print(ts_years)
        print(tr_X.head())
        print(tr_y)
        if w is None:
            tr_w = None
        else:
            tr_w = w[tr_mask]
        #ts_w = w[ts_mask]
        
        ts_year = ts_years[0]
        
        ts_X, ts_y = limit_range(ts_X, ts_y, ts_year)
        
        # print(tr_idx)
        # print(f"train: {tr_dates.min()} - {tr_dates.max()} #{tr_y.shape[0]}")
        print(f"test: {ts_year} #{ts_y.shape[0]}")
        estimator = clone(estimator)
        estimator.fit(tr_X, tr_y, sample_weight=tr_w, eval_set=[(ts_X, ts_y)],
                      eval_metric=custom_metric)
        
        pr_y = estimator.predict(ts_X)
#         base_act_price = ts_X["feat_actual_price"].values[:, None]
#         pr_price = (pr_y + 1) * base_act_price
#         print(pr_price)
#         pr_price = np.round(pr_price)
#         pr_y = pr_price / base_act_price - 1
        
        # print(ts_X.shape, ts_y.shape)
        # pr_y = np.clip(pr_y, -1.0, None)
        score = get_scores(ts_y, pr_y)
        print(score)
        estimators.append(estimator)
        scores.append(score)
        
    return estimators, scores


# estimators, scores = cv(estimator, X, y, None)