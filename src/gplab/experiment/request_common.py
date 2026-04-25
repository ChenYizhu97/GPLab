from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Optional

from gplab.utils.validation import (
    validate_dataset_value,
    validate_model_type_value,
    validate_pool_ratio_value,
    validate_pool_value,
)


@dataclass(frozen=True)
class TrainRequestContext:
    conf: dict
    log_file: Optional[str]
    seed_mode: str
    seed_base: Optional[int]
    allow_duplicate_seeds: bool
    seed_list: Optional[list[int]]


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
    validate_dataset_value(dataset_name)
    validate_pool_ratio_value(pool_ratio)
    validate_model_type_value(model_type)
    is_custom_pool = validate_pool_value(pool)

    if "model" not in model_conf:
        raise ValueError("Missing [model] section in model config")
    if "experiment" not in experiment_conf:
        raise ValueError("Missing [experiment] section in experiment config")

    conf = {
        "model": deepcopy(model_conf["model"]),
        "experiment": deepcopy(experiment_conf["experiment"]),
    }
    expr_conf = conf["experiment"]

    runs = int(expr_conf.get("runs", 0))
    if runs <= 0:
        raise ValueError("Invalid runs value. Require experiment.runs > 0.")

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

    expr_conf["runs"] = runs
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
