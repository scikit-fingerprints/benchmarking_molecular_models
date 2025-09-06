import logging as log
import numpy as np
import os
import joblib
import sys

from hydra.utils import get_original_cwd
from os.path import join
from ..common.types import EmbeddingConfig, Dataset, EmbeddedDataset, Embedder
# from ..common.db import EmbeddingMeta
from time import time


def is_already_embedded(config: EmbeddingConfig, dataset_name: str, model: Embedder) -> bool:
    model_name = model.name
    output_dir = join(get_original_cwd(), config.embedded_directory, dataset_name)
    embedding_file = join(output_dir, f"{model_name}.joblib")
    if sys.version_info < (3, 9):
        return os.path.exists(embedding_file.replace(".joblib", ".json"))
    return os.path.exists(embedding_file)


def clock(config: EmbeddingConfig, dataset: Dataset, model: Embedder) -> None:
    start = time()
    model.embed(dataset.data)
    runtime = time() - start
    return runtime


def embed(config: EmbeddingConfig, dataset: Dataset, model: Embedder, cache: bool = True) -> None:
    model_name = model.name
    output_dir = join(get_original_cwd(), config.embedded_directory, dataset.name)
    embedding_file = join(output_dir, f"{model_name}.joblib" if sys.version_info >= (3, 9) else f"{model_name}.json")
    log_header = f"Model : {model_name: <20}, dataset: {dataset.name: <20} |"
    
    if cache and os.path.exists(embedding_file):
        log.info(f"Loading cached embeddings {embedding_file}")
        return joblib.load(embedding_file)
    elif not cache:
        log.info("Cache disabled, recomputing embeddings")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start = time()
    embeddings = model.embed(dataset.data)
    runtime = time() - start
    
    log.info(f"{log_header} took {runtime:.2f} seconds")

    embedded_dataset = EmbeddedDataset(
        name=dataset.name,
        task=dataset.task,
        embedder=model_name,
        splits=dataset.splits,
        X=embeddings,
        y=dataset.labels
    )
    
    invalid_embeds = embedded_dataset.remove_failed_embeddings()
    if invalid_embeds > 0:
        log.warning(f"{log_header} removed {invalid_embeds} invalid embeddings from the dataset '{dataset.name}'")
        if invalid_embeds > config.max_invalid_embeddings:
            log.error(f"{log_header} too many invalid embeddings ({invalid_embeds}), stopping embedding process")
            raise ValueError(f"Too many invalid embeddings ({invalid_embeds}) for dataset '{dataset.name}' with model '{model_name}'")
    

    if sys.version_info < (3, 9):
        embedded_dataset.serialize_legacy(embedding_file)
    else:
        joblib.dump(embedded_dataset, embedding_file)

    # EmbeddingMeta.create(
    #     dataset=dataset.name,
    #     embedder=model_name,
    #     embedding_loc=embedding_file,
    #     embedding_time=runtime,    
    # )
    
    log.info(f"{log_header} saved results")
    
    del embeddings, embedded_dataset, model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def load_embedding(config: EmbeddingConfig, dataset_name: str, model_name: str) -> EmbeddedDataset:
    output_dir = join(get_original_cwd(), config.embedded_directory, dataset_name)
    embedding_file = join(output_dir, f"{model_name}.joblib")
    return joblib.load(embedding_file)
