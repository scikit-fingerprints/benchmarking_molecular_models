import joblib
import os
import logging as log
import numpy as np

from omegaconf import DictConfig
from os.path import join
from hydra.utils import get_original_cwd
from typing import Optional, Tuple

from ...common.types import EmbeddedDataset


def get_data(dataset: EmbeddedDataset) -> Tuple[np.ndarray, np.ndarray]:
    return dataset.X, dataset.y_np


def get_train_data(dataset: EmbeddedDataset) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset.splits["train"], list):
        train_split = dataset.splits["train"] + dataset.splits["valid"]
    else:
        train_split = dataset.splits["train"].tolist() + dataset.splits["valid"].tolist()
    return dataset.X[train_split], dataset.y_np[train_split]


def get_test_data(dataset: EmbeddedDataset) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset.splits["test"], list):
        test_split = dataset.splits["test"]
    else:
        test_split = dataset.splits["test"].tolist()
    return dataset.X[test_split], dataset.y_np[test_split]


def load_embedding(
    dataset_info: DictConfig,
    model_name: str,
    embedded_dir: str
) -> Optional[EmbeddedDataset]:
    embedded_filename = join(get_original_cwd(), embedded_dir, dataset_info.name, f"{model_name}.joblib")
    legacy_filename = join(get_original_cwd(), embedded_dir, dataset_info.name, f"{model_name}.json")
    
    if os.path.exists(legacy_filename):
        log.info("Legacy embedded dataset found, converting to new format")
        embedded_data = EmbeddedDataset.deserialize_legacy(legacy_filename)
    elif not os.path.exists(embedded_filename):
        log.error(f"Embedded dataset not found: {embedded_filename}")
        # Cannot raise an error, this will stop hydra execution
        return None
    else:
        embedded_data: EmbeddedDataset = joblib.load(embedded_filename)
    
    if embedded_data.X is None:
        log.error("Embedded dataset is empty")
        raise RuntimeError("Embedded dataset is empty")
    
    if len(embedded_data.X.shape) == 1:
        log.warning("Invalid X shape (1 dim), assuming invalid concatenation")
        desired_samples = embedded_data.y.shape[0]
        embedded_data.X = embedded_data.X.reshape(desired_samples, -1)
        
    return embedded_data