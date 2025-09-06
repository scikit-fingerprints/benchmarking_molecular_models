from omegaconf import DictConfig
import hydra
import logging as log

from src.common.types import EmbeddingConfig
from src.eval import eval_procedure, AVAILABLE_HEADS
from src.common.db import DbContex


def eval(cfg, embed_config, model_name, dataset_info,
         short_model_name, model_head, override):
    log.info(f"Evaluating model {model_name} on dataset {dataset_info.name} with metric {dataset_info.metric} with head {model_head}")
    if "safe" in cfg and cfg.safe:
        log.info("Running in safe mode")
        try:
            eval_procedure(
                dataset_info=dataset_info,
                embedded_dir=embed_config.embedded_directory,
                predictions_dir=embed_config.predictions_directory,
                model_name=short_model_name,
                model_head=model_head,
                override=override,
            )
        except Exception as e:
            import traceback
            log.error(f"Error during evaluation: {e}")
            log.error(traceback.format_exc())
            return
    else:
        eval_procedure(
            dataset_info=dataset_info,
            embedded_dir=embed_config.embedded_directory,
            predictions_dir=embed_config.predictions_directory,
            model_name=short_model_name,
            model_head=model_head,
            override=override,
        )


@hydra.main(config_path="./config", config_name="score")
def main(cfg: DictConfig):
    embed_config = EmbeddingConfig(**cfg.embedding)
    if "model" in cfg and "embedding_name" in cfg.model:
        model_name = cfg.model.embedding_name
    elif "model_name" in cfg:
        model_name = cfg.model_name
    else:
        model_name = cfg.model.model_name
    if "gpt" in model_name.lower():
        short_model_name = model_name.split("/")[-1]
    else:
        short_model_name = model_name.split("/")[-1].split('.')[0]
    override = not cfg.cache if "cache" in cfg else False
    print(f"Override status {override}")

    with DbContex(embed_config):
        for model_head in AVAILABLE_HEADS:
            eval(cfg, embed_config, model_name, cfg.dataset,
                 short_model_name, model_head, override)



if __name__ == '__main__':
    main()
    print("All done")
