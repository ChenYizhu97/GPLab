from copy import deepcopy
from typing import Optional

import typer

from utils.cli import (
    resolve_seed_options,
    validate_dataset,
    validate_model_type,
    validate_pool,
    validate_pool_ratio,
)
def _validate_config_sections(model_conf: dict, experiment_conf: dict) -> None:
    if "model" not in model_conf:
        raise ValueError("Missing [model] section in model config")
    if "experiment" not in experiment_conf:
        raise ValueError("Missing [experiment] section in experiment config")


def _normalize_split_ratio(expr_conf: dict) -> tuple[float, float]:
    train_ratio = float(expr_conf.get("train_ratio", 0.8))
    val_ratio = float(expr_conf.get("val_ratio", 0.1))
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Invalid split ratio. Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )
    return train_ratio, val_ratio


def _normalize_runs(expr_conf: dict) -> int:
    runs = int(expr_conf.get("runs", 0))
    if runs <= 0:
        raise ValueError("Invalid runs value. Require experiment.runs > 0.")
    return runs


def apply_job_overrides(model_conf: dict, experiment_conf: dict, job: dict) -> tuple[dict, dict]:
    merged_model = deepcopy(model_conf)
    merged_experiment = deepcopy(experiment_conf)

    model_section = merged_model.setdefault("model", {})
    experiment_section = merged_experiment.setdefault("experiment", {})

    if "model" in job:
        model_section.update(job["model"])

    if "train" in job:
        train = job["train"]
        for field in ("runs", "lr", "batch_size", "patience", "epochs", "train_ratio", "val_ratio"):
            if field in train:
                experiment_section[field] = train[field]

        for field in ("seed_mode", "seed_base", "seed_list", "allow_duplicate_seeds"):
            if field in train:
                experiment_section[field] = train[field]

    return merged_model, merged_experiment


def build_experiment_config(
    model_conf: dict,
    experiment_conf: dict,
    *,
    pool: str,
    pool_ratio: float,
    dataset_name: str,
    model_type: str,
    tag: Optional[str],
    seed_mode: str,
    seed_base: Optional[int],
    seed_list: Optional[list[int]],
    allow_duplicate_seeds: bool,
) -> dict:
    validate_dataset(dataset_name)
    validate_pool_ratio(pool_ratio)
    validate_model_type(model_type)
    is_custom_pool = validate_pool(pool)

    _validate_config_sections(model_conf, experiment_conf)

    conf = {
        "model": deepcopy(model_conf["model"]),
        "experiment": deepcopy(experiment_conf["experiment"]),
    }
    expr_conf = conf["experiment"]
    expr_conf["runs"] = _normalize_runs(expr_conf)
    train_ratio, val_ratio = _normalize_split_ratio(expr_conf)

    expr_conf["train_ratio"] = train_ratio
    expr_conf["val_ratio"] = val_ratio
    expr_conf["seed_mode"] = seed_mode
    expr_conf["seed_base"] = seed_base
    expr_conf["seed_list"] = seed_list
    expr_conf["allow_duplicate_seeds"] = allow_duplicate_seeds
    conf["model"]["variant"] = model_type

    conf["pool"] = {
        "method": pool,
        "ratio": pool_ratio,
        "source": "custom_factory" if is_custom_pool else "builtin",
    }
    conf["dataset"] = dataset_name
    if tag is not None:
        conf["tag"] = tag

    return conf


def build_request_from_sources(
    *,
    model_conf: dict,
    experiment_conf: dict,
    job: Optional[dict],
    pool: Optional[str],
    pool_ratio: Optional[float],
    dataset_name: Optional[str],
    model_type: Optional[str],
    tag: Optional[str],
    log_file: Optional[str],
    seed_mode: Optional[str],
    seed_base: Optional[int],
    seed_list: Optional[str],
    allow_duplicate_seeds: Optional[bool],
) -> tuple[dict, Optional[str], str, Optional[int], bool, Optional[list[int]]]:
    merged_model_conf = deepcopy(model_conf)
    merged_experiment_conf = deepcopy(experiment_conf)

    if job is not None:
        merged_model_conf, merged_experiment_conf = apply_job_overrides(
            merged_model_conf,
            merged_experiment_conf,
            job,
        )

    job_pool = (job or {}).get("pool", {})
    job_model = (job or {}).get("model", {})

    final_pool = pool if pool is not None else job_pool.get("name", "nopool")
    final_pool_ratio = pool_ratio if pool_ratio is not None else job_pool.get("ratio", 0.5)
    final_dataset = dataset_name if dataset_name is not None else (job or {}).get("dataset", "PROTEINS")
    final_model_type = model_type if model_type is not None else job_model.get("variant", "sum")
    final_tag = tag if tag is not None else (job or {}).get("tag")
    final_log_file = log_file if log_file is not None else (job or {}).get("log_file")

    exp = merged_experiment_conf.get("experiment", {})
    try:
        final_seed_mode, final_seed_base, final_allow_dup, final_seed_list = resolve_seed_options(
            seed_mode=seed_mode,
            seed_base=seed_base,
            seed_list=seed_list,
            allow_duplicate_seeds=allow_duplicate_seeds,
            expr_conf=exp,
        )
    except typer.BadParameter:
        raise
    except Exception as exc:  # pragma: no cover - defensive bridge
        raise ValueError(str(exc)) from exc

    conf = build_experiment_config(
        model_conf=merged_model_conf,
        experiment_conf=merged_experiment_conf,
        pool=final_pool,
        pool_ratio=float(final_pool_ratio),
        dataset_name=final_dataset,
        model_type=final_model_type,
        tag=final_tag,
        seed_mode=final_seed_mode,
        seed_base=final_seed_base,
        seed_list=final_seed_list,
        allow_duplicate_seeds=final_allow_dup,
    )
    return conf, final_log_file, final_seed_mode, final_seed_base, final_allow_dup, final_seed_list


def build_request_from_normalized_job(job: dict) -> tuple[dict, Optional[str], str, int, bool, Optional[list[int]]]:
    normalized = deepcopy(job)
    model_conf = {"model": deepcopy(normalized["model"])}
    experiment_conf = {
        "experiment": {
            "runs": normalized["train"]["runs"],
            "lr": normalized["train"]["lr"],
            "batch_size": normalized["train"]["batch_size"],
            "patience": normalized["train"]["patience"],
            "epochs": normalized["train"]["epochs"],
            "train_ratio": normalized["train"]["train_ratio"],
            "val_ratio": normalized["train"]["val_ratio"],
            "seed_mode": normalized["train"]["seed_mode"],
            "seed_base": normalized["train"]["seed_base"],
            "seed_list": deepcopy(normalized["train"]["seed_list"]),
            "allow_duplicate_seeds": normalized["train"]["allow_duplicate_seeds"],
        }
    }
    conf = build_experiment_config(
        model_conf=model_conf,
        experiment_conf=experiment_conf,
        pool=normalized["pool"]["name"],
        pool_ratio=normalized["pool"]["ratio"],
        dataset_name=normalized["dataset"],
        model_type=normalized["model"]["variant"],
        tag=normalized["tag"],
        seed_mode=normalized["train"]["seed_mode"],
        seed_base=normalized["train"]["seed_base"],
        seed_list=deepcopy(normalized["train"]["seed_list"]),
        allow_duplicate_seeds=normalized["train"]["allow_duplicate_seeds"],
    )
    return (
        conf,
        normalized["log_file"],
        normalized["train"]["seed_mode"],
        normalized["train"]["seed_base"],
        normalized["train"]["allow_duplicate_seeds"],
        deepcopy(normalized["train"]["seed_list"]),
    )
