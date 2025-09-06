import hydra
import os
import joblib

from hydra.utils import get_original_cwd
from src.common.data_v2 import load
from src.common.types import EmbeddingConfig


@hydra.main(config_path="./config", config_name="downloader")
def main(cfg):
    embed_config = EmbeddingConfig(**cfg.embedding)
    dataset_name = cfg.dataset.name
    destination = os.path.join(get_original_cwd(), embed_config.prepared_directory)

    os.makedirs(destination, exist_ok=True)
    filename = os.path.join(destination, f"{dataset_name}.joblib")
    legacy_filename = os.path.join(destination, f"{dataset_name}.json")

    if (os.path.exists(filename) or os.path.exists(legacy_filename)) and cfg.cache:
        print(f"Dataset {dataset_name} already exists at {filename}")
        return

    dataset = load(cfg.dataset, cfg.embedding.raw_directory)
    
    joblib.dump(dataset, filename)
    dataset.serialize_legacy(legacy_filename)
    print(f"Dataset {dataset_name} saved to {filename}")
    

if __name__ == '__main__':
    main()
