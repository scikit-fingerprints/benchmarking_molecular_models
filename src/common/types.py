from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from abc import ABC, abstractmethod

import logging as log
import pandas as pd
import numpy as np
import json
import base64
import torch


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, torch.Tensor):
            data_b64 = base64.b64encode(obj.cpu().numpy().tobytes()).decode('utf-8')
            return dict(__torch_tensor__=data_b64,
                        dtype=str(obj.dtype).split('.')[-1],
                        shape=obj.shape)
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(np.ascontiguousarray(obj).data).decode('utf-8')
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        elif isinstance(obj, pd.DataFrame):
            return dict(__dataframe__=obj.to_dict(orient='split'))
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray or pandas DataFrame
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray or DataFrame
    :return: (ndarray or DataFrame) if input was an encoded ndarray or DataFrame
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    elif isinstance(dct, dict) and '__dataframe__' in dct:
        return pd.DataFrame(**dct['__dataframe__'])
    elif isinstance(dct, dict) and '__torch_tensor__' in dct:
        data = base64.b64decode(dct['__torch_tensor__'])
        return torch.tensor(np.frombuffer(data, dct['dtype']).reshape(dct['shape']))
    return dct


@dataclass
class EmbeddingConfig:
    raw_directory: str
    embedded_directory: str
    predictions_directory: str
    data_directory: str
    clock_directory: str
    database: str
    prepared_directory: str
    svd_directory: str
    max_invalid_embeddings: int
    max_samples: Optional[int] = None


@dataclass
class SystemConfig:
    embedding_config: EmbeddingConfig
    
    
@dataclass
class Dataset:
    name: str
    task: Literal['classification', 'regression']
    data: Any
    splits: Any

    def filter_out_problematic_molecules(self, illegal_smiles: List[str]):
        """
        Filters out rows from the dataset where the 'smiles' column contains any of the illegal SMILES strings.
        :param illegal_smiles: List of illegal SMILES strings to filter out.
        """
        if 'smiles' not in self.data.columns:
            raise ValueError("Dataset does not contain a 'smiles' column.")
        
        if 'split' not in self.data.columns:
            log.warning("Dataset does not contain split in dataframe, fixing")
            self.data['split'] = 'UNKNOWN'
            for k, indices in self.splits.items():
                idx_list = list(indices) if not isinstance(indices, list) else indices
                if len(idx_list) > 0:
                    self.data.iloc[idx_list, self.data.columns.get_loc('split')] = k
        
        initial_count = len(self.data)
        self.data = self.data[~self.data['smiles'].isin(illegal_smiles)]
        filtered_count = len(self.data)
        
        log.info(f"Filtered out {initial_count - filtered_count} problematic molecules from dataset '{self.name}'.")
        log.info("Fixing splits indices")
        self.splits = {
            k: np.where(self.data['split'] == k)[0].tolist()
            for k in self.splits.keys()
        }
        if 'UNKNOWN' in self.splits:
            log.error("Something went really wrong.... ")
            raise ValueError("Dataset contains 'UNKNOWN' split, which should not happen.")
        
    
    @property
    def labels(self) -> pd.DataFrame:
        id_cols = [x for x in self.data.columns if 'id' in x.lower() or 'split' in x.lower()]
        
        return self.data.drop(columns=(['smiles', 'graph'] + id_cols))

    def serialize_legacy(self, path):
        import json
        obj = {
            "name": self.name,
            "task": self.task,
            "data": self.data,
            "splits": self.splits
        }
        with open(path, 'w') as f:
            json.dump(obj, f, cls=NumpyEncoder)

    @classmethod
    def deserialize_legacy(cls, path):
        import json
        with open(path, 'r') as f:
            return cls(**json.load(f, object_hook=json_numpy_obj_hook))


@dataclass
class EmbeddedDataset:
    name: str
    task: Literal['classification', 'regression']
    embedder: str
    splits: Any
    X: np.ndarray
    y: pd.DataFrame
    
    def remove_failed_embeddings(self) -> int:
        """Performs validation of missing values, drop failed molecules and return number of invalid samples.

        Returns:
            int: number of invalid samples
        """
        print(f"Shape of X: {self.X.shape}")
        if len(self.X.shape) == 1:
            log.warning(f"Dataset '{self.name}' has only one dimension in X, reshaping to 2D.")
            self.X = self.X.reshape(self.y.shape[0], -1)
        valid_indices = np.where(~np.isnan(self.X).any(axis=1))[0]
        invalid_count = self.X.shape[0] - valid_indices.shape[0]
        split_to_idx = {k: i for i, k in enumerate(self.splits.keys())}
        if invalid_count == 0:
            return 0
        
        log.warning(f"Found {invalid_count} invalid samples in the dataset '{self.name}'.")
        splits_raw = -1 * np.ones(self.X.shape[0], dtype=int)
        for split_name, indices in self.splits.items():
            splits_raw[indices] = split_to_idx[split_name]
        self.X = self.X[valid_indices]
        self.y = self.y.iloc[valid_indices]
        splits_raw = splits_raw[valid_indices]
        
        self.splits = {k: np.where(splits_raw == i)[0].tolist() for k, i in split_to_idx.items()}
        log.info(f"Removed {invalid_count} invalid samples from the dataset '{self.name}'")
        return invalid_count
        

    @property
    def y_np(self) -> np.ndarray:
        if 'split' in self.y.columns:
            self.y = self.y.drop(columns=['split'])
        
        arr = self.y.to_numpy()
        if arr.shape[1] == 1:
            return arr.flatten()
        return arr

    def serialize_legacy(self, path):
        import json
        obj = {
            "name": self.name,
            "task": self.task,
            "embedder": self.embedder,
            "splits": self.splits,
            "X": self.X,
            "y": self.y
        }
        with open(path, 'w') as f:
            json.dump(obj, f, cls=NumpyEncoder)

    @classmethod
    def deserialize_legacy(cls, path):
        import json
        with open(path, 'r') as f:
            return cls(**json.load(f, object_hook=json_numpy_obj_hook))

    
@dataclass
class HeadResult:
    embedder: str
    dataset_name: str
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    cv_score: float
    model: str
    hyperparams: Dict[str, Any]


@dataclass
class EvaluationResult:
    embedder: str
    metric_name: str
    metric_value: float
    cv_metric_value: float
    model: str
    hyperparams: Dict[str, Any]    


class Embedder(ABC):
    @abstractmethod
    def embed(self, data):
        pass
    
    @property
    def name(self):
        return type(self).__name__
    
    @property
    def device_used(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    

class SmilesEmbedder(Embedder):
    @abstractmethod
    def forward(self, smiles):
        pass
    
    def embed(self, data):
        nulls = data["smiles"].isna().sum()
        if nulls > 0:
            raise ValueError(f"Input data contains {nulls} null SMILES strings, which cannot be processed.")
        illegal_smiles = data["smiles"].str.contains("*", regex=False).sum()
        if illegal_smiles > 0:
            log.warning(f"Input data contains {illegal_smiles} SMILES strings with illegal characters ('*'). Replacing wildcards with 'C'.")
            data["smiles"] = data["smiles"].str.replace("*", "C", regex=False)
        
        return self.forward(data["smiles"])


class GraphEmbedder(Embedder):
    @abstractmethod
    def forward(self, graphs):
        pass
    
    def embed(self, data):
        return self.forward(data["graph"])
