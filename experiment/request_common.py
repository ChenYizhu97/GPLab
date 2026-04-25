from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Optional

from utils.cli import (
    resolve_seed_options,
    validate_dataset,
    validate_model_type,
    validate_pool,
    validate_pool_ratio,
)


@dataclass(frozen=True)
class TrainRequestContext:
    conf: dict
    log_file: Optional[str]
    seed_mode: str
    seed_base: Optional[int]
    allow_duplicate_seeds: bool
    seed_list: Optional[list[int]]


def validate_config_sections(model_conf: dict, experiment_conf: dict) -> None:
    if "model" not in model_conf:
        raise ValueError("Missing [model] section in model config")
    if "experiment" not in experiment_conf:
        raise ValueError("Missing [experiment] section in experiment config")


def normalize_split_ratio(expr_conf: dict) -> tuple[float, float]:
    train_ratio = float(expr_conf.get("train_ratio", 0.8))
    val_ratio = float(expr_conf.get("val_ratio", 0.1))
    if (
        not math.isfinite(train_ratio)
        or not math.isfinite(val_ratio)
        or train_ratio <= 0
        or val_ratio <= 0
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError(
            "Invalid split ratio. Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )
    return train_ratio, val_ratio


def normalize_runs(expr_conf: dict) -> int:
    runs = int(expr_conf.get("runs", 0))
    if runs <= 0:
        raise ValueError("Invalid runs value. Require experiment.runs > 0.")
    return runs


def build_internal_request(
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

    validate_config_sections(model_conf, experiment_conf)

    conf = {
        "model": deepcopy(model_conf["model"]),
        "experiment": deepcopy(experiment_conf["experiment"]),
    }
    expr_conf = conf["experiment"]
    expr_conf["runs"] = normalize_runs(expr_conf)
    train_ratio, val_ratio = normalize_split_ratio(expr_conf)

    expr_conf["train_ratio"] = train_ratio
    expr_conf["val_ratio"] = val_ratio
    expr_conf["seed_mode"] = seed_mode
    expr_conf["seed_base"] = seed_base
    expr_conf["seed_list"] = deepcopy(seed_list)
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


def resolve_request_seed_options(
    *,
    seed_mode: Optional[str],
    seed_base: Optional[int],
    seed_list: Optional[str],
    allow_duplicate_seeds: Optional[bool],
    expr_conf: dict,
) -> tuple[str, int, bool, Optional[list[int]]]:
    return resolve_seed_options(
        seed_mode=seed_mode,
        seed_base=seed_base,
        seed_list=seed_list,
        allow_duplicate_seeds=allow_duplicate_seeds,
        expr_conf=expr_conf,
    )
