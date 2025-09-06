import numpy as np
import json

from .models import RF_CLF, RF_REG, RIDGE_CLF, RIDGE_REG
from sklearn.metrics import get_scorer, get_scorer_names, make_scorer, precision_recall_curve, auc, mean_absolute_error
from scipy.stats import spearmanr


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def spearman_corr(x, y):
    rho, _ = spearmanr(x, y)
    return rho


def pr_auc(x, y):
    p, r, _ = precision_recall_curve(x, y)
    return auc(r, p)


def get_model_version_hash() -> str:
    params_string = json.dumps({
        "RF_CLF": RF_CLF,
        "RF_REG": RF_REG,
        "RIDGE_CLF": RIDGE_CLF,
        "RIDGE_REG": RIDGE_REG,
    }, sort_keys=True, cls=NpEncoder)
    return str(hash(params_string))


def get_sklearn_scorer(metric_name: str):
    if metric_name in get_scorer_names():
        return get_scorer(metric_name)
    
    if metric_name == "spearmancorr":
        return make_scorer(
            spearman_corr,
            greater_is_better=True
        )
    
    if metric_name == "pr_auc_score":
        return make_scorer(
            pr_auc,
            greater_is_better=True
        )
        
    if metric_name == "mae":
        return make_scorer(
            mean_absolute_error,
            greater_is_better=False
        )
        
    raise ValueError(f"Unknown metric name: {metric_name}")