from scipy.stats import spearmanr


def custom_metric(t, y):
    from scipy.stats import spearmanr
    score = spearmanr(t, y, nan_policy="propagate")[0]
    return 'rho', score, True
