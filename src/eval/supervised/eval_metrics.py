import os
import numpy as np
import logging as log

from hydra.utils import get_original_cwd
from typing import Tuple
from tdc.benchmark_group import admet_group
from ogb.graphproppred import Evaluator
from ...common.types import HeadResult, EvaluationResult
from .utils import get_sklearn_scorer
from skfp.metrics import multioutput_auroc_score
from sklearn.metrics import roc_auc_score


def evaluate_tdc(y_pred: np.ndarray, dataset_name: str) -> Tuple[str, float]:
    """
    Returns:
        Tuple[str, float]: Tuple containing metric name and value
    """
    grp_path = os.path.join(get_original_cwd(), "data/cache")
    os.makedirs(grp_path, exist_ok=True)
    group = admet_group(path=grp_path)
    benchmark = group.get(dataset_name)
    
    predictions = {
        benchmark["name"]: y_pred
    }
    
    metric_name, metric_value = list(list(group.evaluate(predictions).values())[0].items())[0]
    return metric_name, metric_value


def evaluate_ogb(y_pred: np.ndarray, y_test: np.ndarray, dataset_name: str) -> Tuple[str, float]:
    """
    Returns:
        Tuple[str, float]: Tuple containing metric name and value
    """
    evaluator = Evaluator(name=dataset_name)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    
    input_dict = {
        "y_true": y_test,
        "y_pred": y_pred
    }
    result_dict = evaluator.eval(input_dict)
    return list(result_dict.items())[0]


def evaluate_sklearn(y_pred: np.ndarray, y_test: np.ndarray, metric_name: str) -> Tuple[str, float]:
    scorer = get_sklearn_scorer(metric_name)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    metric_value = scorer._score_func(y_test, y_pred)
    return metric_name, metric_value


def log_predictions(data: HeadResult, pred_directory: str):
    print(f"Logging predictions for {data.dataset_name} dataset")
    res_path = os.path.join(get_original_cwd(), pred_directory, data.dataset_name, data.embedder, f"{data.model}.npy")
    print(f"Saving predictions to {res_path}")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    np.save(res_path, data.y_test_pred)


def evaluate_tdc_safe(y_pred: np.ndarray, y_test: np.ndarray, dataset_name: str, fallback_metric: str) -> Tuple[str, float]:
    """
    Safely evaluates TDC datasets, falling back to a specified metric if the dataset is not found.
    
    Returns:
        Tuple[str, float]: Tuple containing metric name and value
    """
    try:
        return evaluate_tdc(y_pred, dataset_name)
    except Exception as e:
        log.error(f"TDC failed for dataset {dataset_name}. Falling back to {fallback_metric}. Cause: {e}")
        return evaluate_sklearn(y_pred, y_test, fallback_metric)


def get_skfp_roc_auc(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    
    if y_test.ndim == 1:
        y_pred = y_pred[:, 1]
    # In multioutput case, the prediction are in shape (n_class, n_samples, 2), and we want (n_samples, n_class) for positive label
    if y_pred.ndim == 3:
        y_pred = y_pred[:, :, 1].T
        
    if np.isnan(np.min(y_test)):
        return multioutput_auroc_score(y_test, y_pred)
    try:
        return roc_auc_score(y_test, y_pred)
    except Exception:
        return multioutput_auroc_score(y_test, y_pred)


def evaluate(data: HeadResult, dataset_config, pred_directory: str) -> EvaluationResult:
    y_pred = data.y_test_pred
    y_test = data.y_test_true
    
    log_predictions(data, pred_directory)
    
    metric_name = dataset_config.metric
    metric_value = get_skfp_roc_auc(y_pred, y_test)
    
    return EvaluationResult(
        embedder=data.embedder,
        metric_name=metric_name,
        metric_value=metric_value,
        model=data.model,
        hyperparams=data.hyperparams,
        cv_metric_value=data.cv_score
    )
