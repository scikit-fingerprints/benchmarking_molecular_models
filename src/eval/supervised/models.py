import numpy as np
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import  Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from skfp.distances import tanimoto_count_distance


RF_CLF = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["entropy"],
}

RF_REG = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["squared_error"],
}

RIDGE__MULTIOUTPUT_CLF = {
    "clf__estimator__C": 1 / np.logspace(-2, 3, 10),
    "clf__estimator__penalty": ["l2"],
    "clf__estimator__solver": ["lbfgs"],
    "clf__estimator__max_iter": [5000],
}

RIDGE_CLF = {
    "clf__C": 1 / np.logspace(-2, 3, 10),
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"],
    "clf__max_iter": [5000],
}


RIDGE_REG = {
    "clf__alpha": np.logspace(-2, 3, 10),
    "clf__max_iter": [5000],
    "clf__solver": ["lbfgs"],
}

KNN_CLF = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}

KNN_REG = {
    "clf__n_neighbors": np.arange(1, 11, 2),
}


AVAILABLE_HEADS = ["rf", "ridge", "knn"]


def get_knn_distance(embeddings_dtype):
    if np.issubdtype(embeddings_dtype, np.integer):
        return tanimoto_count_distance
    elif np.issubdtype(embeddings_dtype, np.floating):
        return "cosine"
    else:
        raise ValueError(f"Unsupported embeddings dtype: {embeddings_dtype}. Expected integer or floating point type.")


def get_clf_models(no_output: int, embeddings_dtype):
    if no_output == 1:
        lr_clf = LogisticRegression(n_jobs=-1)
        lr_params = RIDGE_CLF
    else:
        lr_clf = MultiOutputClassifier(LogisticRegression(n_jobs=-1))
        lr_params = RIDGE__MULTIOUTPUT_CLF
    
    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestClassifier(n_jobs=-1))]),
            "params": RF_CLF.copy(),
        },
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", lr_clf),
                ]
            ),
            "params": lr_params.copy(),
        },
        "knn": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_jobs=-1, metric=get_knn_distance(embeddings_dtype))),
                ]
            ),
            "params": KNN_CLF.copy(),
        }
    }


def get_reg_models(embeddings_dtype):
    return {
        "rf": {
            "model": Pipeline([("clf", RandomForestRegressor(n_jobs=-1))]),
            "params": RF_REG.copy(),
        },
        "ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", Ridge()),
                ]
            ),
            "params": RIDGE_REG.copy(),
        },
        "knn": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsRegressor(n_jobs=-1, metric=get_knn_distance(embeddings_dtype))),
                ]
            ),
            "params": KNN_REG.copy(),
        }
    }
