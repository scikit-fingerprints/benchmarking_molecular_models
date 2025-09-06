# =============================================================================
# WARNING: Do not call directly, only use via bash scripts
# =============================================================================

import joblib
import os
import sys
import hydra
import logging as log

from time import sleep
from os.path import join  # noqa: E402
from hydra.utils import get_original_cwd # noqa: E402
from src.common.types import EmbeddingConfig, Dataset # noqa: E402
from src.embedding.embedding import embed, is_already_embedded # noqa: E402
from wrapper import get_embedder


@hydra.main(config_path="./config", config_name="embed")
def main(cfg):
    try:
        from rdkit import RDLogger  
        RDLogger.DisableLog('rdApp.*')    
    except ImportError:
        pass
    embed_config = EmbeddingConfig(**cfg.embedding)
    if "model_name" in cfg:
        model_name = cfg.model_name
    else:
        model_name = cfg.model.model_name

    kwargs = cfg.model.kwargs if "model" in cfg and "kwargs" in cfg.model else {}

    model = get_embedder(model_name, task = cfg.dataset.task, **kwargs)
    log.info(f"Embedding model: {model.name}")
    dataset_name = cfg.dataset.name
    
    if is_already_embedded(embed_config, dataset_name, model) and cfg.cache:
        log.info(f"Embedding already exists for {model.name} on {dataset_name}")
        return
    
    dataset_path = os.path.join(get_original_cwd(), embed_config.prepared_directory, f"{dataset_name}.joblib")
    # if python < 3.9 use legacy loader
    if sys.version_info < (3, 9):
        print("Using legacy loader")
        dataset = Dataset.deserialize_legacy(dataset_path.replace(".joblib", ".json"))
    else:
        print("Using joblib loader")
        dataset = joblib.load(dataset_path)
        
    # illegal_smiles = cfg.model.illegal_smiles if 'model' in cfg and "illegal_smiles" in cfg.model else None
    illegal_smiles_path = cfg.illegal_smiles if 'illegal_smiles' in cfg else None
    if illegal_smiles_path is not None:
        with open(join(get_original_cwd(), illegal_smiles_path), 'r') as f:
            illegal_smiles = [line.strip() for line in f.readlines()]
        log.info(f"Filtering out illegal SMILES: {illegal_smiles}")
        dataset.filter_out_problematic_molecules(illegal_smiles)
    
    # with DbContex(cfg.embedding):
    embed(embed_config, dataset, model, cache=cfg.cache)
    sleep(10)


if __name__ == '__main__':
    main()
