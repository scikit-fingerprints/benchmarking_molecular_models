import os

import joblib
import json
import torch
import logging as log

from omegaconf import DictConfig
from os.path import join
from hydra.utils import get_original_cwd

from .const import DEFAULT_MEMORY_WEIGHT
from .train import fit_and_eval_embedding
from .eval_metrics import evaluate
from .utils import get_model_version_hash, NpEncoder
from ...common.types import EvaluationResult, EmbeddedDataset
from ...common.db import ClassificationReport


def eval_embedding(
    data: EmbeddedDataset,
    pred_directory: str,
    dataset_config,
    metric_name: str,
    model_head: str,
) -> EvaluationResult:
    log.info("Training model")
    head_result = fit_and_eval_embedding(
        dataset=data, 
        metric_name=metric_name, 
        model_head=model_head,
        memory_weight=dataset_config.get('memory_weight', DEFAULT_MEMORY_WEIGHT))
    log.info(f"Training complete, best CV result: {head_result.cv_score}")
    return evaluate(head_result, dataset_config, pred_directory)


def dump_hyperparams(hyperparams: dict) -> str:
    return json.dumps(hyperparams, sort_keys=True, cls=NpEncoder)


def check_if_already_evaluated(
        dataset_name: str,
        model_name: str,
        metric_name: str,
        head_name: str,
) -> bool:
    runs = ClassificationReport.select().where(
        (ClassificationReport.dataset == dataset_name) &
        (ClassificationReport.embedder == model_name) &
        (ClassificationReport.cv_metric_name == metric_name) &
        (ClassificationReport.model == head_name)
    ).count()
    if runs > 1:
        raise ValueError("Multiple runs found for the same model, dataset and metric")
    return runs == 1


def delete_previous_evaluations(
        dataset_name: str,
        model_name: str,
        metric_name: str,
        head_name: str,
):
    ClassificationReport.delete().where(
        (ClassificationReport.dataset == dataset_name) &
        (ClassificationReport.embedder == model_name) &
        (ClassificationReport.cv_metric_name == metric_name) &
        (ClassificationReport.model == head_name)
        # (ClassificationReport.library_hash == model_version_hash)
    ).execute()
    log.warning(f"Deleted previous evaluations, dataset: {dataset_name}, model: {model_name}, metric: {metric_name}, head: {head_name}")


def eval_procedure(
    dataset_info: DictConfig,
    embedded_dir: str,
    predictions_dir: str,
    model_name: str,
    model_head: str,
    override: bool = False,
):
    model_version_hash = get_model_version_hash()

    if check_if_already_evaluated(dataset_info.name, model_name,
                                  dataset_info.metric,
                                  model_head):
        if not override:
            log.info("Model already evaluated, skipping")
            return
        log.warning("Model already evaluated, overriding")
        delete_previous_evaluations(dataset_info.name, model_name, dataset_info.metric, model_head)
        
    if model_head == 'knn' and 'muv' in dataset_info.name:
        log.error("Skipping KNN evaluation for MUV datasets, not supported")
        return

    embedded_filename = join(get_original_cwd(), embedded_dir, dataset_info.name, f"{model_name}.joblib")
    legacy_filename = join(get_original_cwd(), embedded_dir, dataset_info.name, f"{model_name}.json")
    
    if os.path.exists(legacy_filename):
        log.info("Legacy embedded dataset found, converting to new format")
        embedded_data = EmbeddedDataset.deserialize_legacy(legacy_filename)
    elif not os.path.exists(embedded_filename):
        log.error(f"Embedded dataset not found: {embedded_filename}")
        # Cannot raise an error, this will stop hydra execution
        return
    else:
        embedded_data: EmbeddedDataset = joblib.load(embedded_filename)

    if embedded_data.X is None:
        log.error("Embedded dataset is empty")
        raise RuntimeError("Embedded dataset is empty")
    
    if isinstance(embedded_data.X, torch.Tensor):
        log.info("Converting torch.Tensor to numpy array")
        embedded_data.X = embedded_data.X.detach().cpu().numpy()
    
    if len(embedded_data.X.shape) == 1:
        log.warning("Invalid X shape (1 dim), assuming invalid concatenation")
        desired_samples = embedded_data.y.shape[0]
        embedded_data.X = embedded_data.X.reshape(desired_samples, -1)
    log.info(f"Shape {embedded_data.X.shape} for dataset {embedded_data.name}, task {embedded_data.task}")

    result = eval_embedding(
        embedded_data,
        predictions_dir,
        dataset_info,
        dataset_info.metric,
        model_head,
    )
    log.info(f"Evaluation complete, test result: {result.metric_value}")
    
    ClassificationReport.create(
        dataset=embedded_data.name,
        task=embedded_data.task,
        embedder=embedded_data.embedder,
        model=result.model,
        hyperparams=dump_hyperparams(result.hyperparams),
        library_hash=model_version_hash,
        cv_metric_name=dataset_info.metric,
        cv_metric=result.cv_metric_value,
        test_metric_name=result.metric_name,
        test_metric=result.metric_value,   
    )