from copy import deepcopy
from typing import Optional

from .defaults import AUTOMATION_MODEL_DEFAULTS, AUTOMATION_TRAIN_DEFAULTS
from .schema import FULL_TRAIN_FIELDS, compute_train_job_case_id, normalize_train_job, require_mapping


def build_case_manifest(
    *,
    pools: list[str],
    datasets: list[str],
    model_types: list[str],
    pool_ratio: float,
    tag_prefix: Optional[str] = None,
    train_overrides: Optional[dict] = None,
    log_file: Optional[str] = None,
) -> list[dict]:
    manifest = []
    for dataset in datasets:
        for pool in pools:
            for model_type in model_types:
                model_block = deepcopy(AUTOMATION_MODEL_DEFAULTS)
                model_block["variant"] = model_type

                train_block = deepcopy(AUTOMATION_TRAIN_DEFAULTS)
                if train_overrides:
                    train_block.update(deepcopy(train_overrides))

                job = normalize_train_job(
                    {
                        "dataset": dataset,
                        "pool": {"name": pool, "ratio": pool_ratio},
                        "model": model_block,
                        "train": train_block,
                        "log_file": log_file,
                        "tag": f"{tag_prefix}_{pool}_{dataset}_{model_type}" if tag_prefix else None,
                    }
                )
                manifest.append(
                    {
                        "case_id": compute_train_job_case_id(job),
                        "dataset": dataset,
                        "pool": pool,
                        "pool_ratio": pool_ratio,
                        "model_type": model_type,
                        "job": job,
                    }
                )
    return manifest


def build_execution_plan_from_configs(
    *,
    model_conf: dict,
    experiment_conf: dict,
    pools: list[str],
    datasets: list[str],
    model_types: list[str],
    pool_ratio: float,
    tag_prefix: Optional[str] = None,
    log_file: Optional[str] = None,
    seed_mode: Optional[str] = None,
    seed_base: Optional[int] = None,
    seed_list: Optional[list[int]] = None,
    allow_duplicate_seeds: Optional[bool] = None,
) -> list[dict]:
    require_mapping(model_conf.get("model"), label="model config")
    require_mapping(experiment_conf.get("experiment"), label="experiment config")
    manifest = []
    for dataset in datasets:
        for pool in pools:
            for model_type in model_types:
                model_block = deepcopy(AUTOMATION_MODEL_DEFAULTS)
                model_block.update(deepcopy(model_conf["model"]))
                model_block["variant"] = model_type

                train_block = deepcopy(AUTOMATION_TRAIN_DEFAULTS)
                train_block.update(
                    {
                        key: deepcopy(value)
                        for key, value in experiment_conf["experiment"].items()
                        if key in FULL_TRAIN_FIELDS
                    }
                )
                if seed_mode is not None:
                    train_block["seed_mode"] = seed_mode
                if seed_base is not None:
                    train_block["seed_base"] = seed_base
                if seed_list is not None:
                    train_block["seed_list"] = deepcopy(seed_list)
                    train_block["seed_mode"] = "list"
                if allow_duplicate_seeds is not None:
                    train_block["allow_duplicate_seeds"] = allow_duplicate_seeds

                job = normalize_train_job(
                    {
                        "dataset": dataset,
                        "pool": {"name": pool, "ratio": pool_ratio},
                        "model": model_block,
                        "train": train_block,
                        "log_file": log_file,
                        "tag": f"{tag_prefix}_{pool}_{dataset}_{model_type}" if tag_prefix else None,
                    }
                )
                manifest.append(
                    {
                        "case_id": compute_train_job_case_id(job),
                        "dataset": dataset,
                        "pool": pool,
                        "pool_ratio": pool_ratio,
                        "model_type": model_type,
                        "job": job,
                    }
                )
    return manifest

