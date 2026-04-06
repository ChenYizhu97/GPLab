import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

from utils.cli import validate_dataset, validate_model_type, validate_pool, validate_pool_ratio


AUTOMATION_MODEL_DEFAULTS = {
    "hidden_features": 128,
    "nonlinearity": "relu",
    "p_dropout": 0.0,
    "conv_layer": "GCN",
    "pre_gnn": [128],
    "post_gnn": [256, 128],
    "variant": "sum",
}

AUTOMATION_TRAIN_DEFAULTS = {
    "runs": 10,
    "lr": 0.0005,
    "batch_size": 32,
    "patience": 50,
    "epochs": 500,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "seed_mode": "auto",
    "seed_base": 20260320,
    "seed_list": None,
    "allow_duplicate_seeds": False,
}

JOB_TOP_LEVEL_FIELDS = {"dataset", "pool", "model", "train", "log_file", "tag"}
JOB_POOL_FIELDS = {"name", "ratio"}
FULL_MODEL_FIELDS = set(AUTOMATION_MODEL_DEFAULTS)
FULL_TRAIN_FIELDS = set(AUTOMATION_TRAIN_DEFAULTS)


def _require_mapping(value, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _reject_unknown_fields(payload: dict, *, allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {label} field(s): {joined}.")


def _require_keys(payload: dict, *, required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required {label} field(s): {joined}.")


def _normalize_seed_list(seed_list) -> Optional[list[int]]:
    if seed_list is None:
        return None
    if not isinstance(seed_list, list) or not seed_list:
        raise ValueError("train.seed_list must be a non-empty array of integers.")
    return [int(value) for value in seed_list]


def _normalize_optional_string(value, *, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null.")
    return value


def _require_string(value, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    return value


def _normalize_bool(value, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _normalize_int(value, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _normalize_float(value, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def load_job_file(path: str) -> dict:
    job_path = Path(path)
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found: {path}")

    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid job JSON: {exc}") from exc

    return _require_mapping(payload, label="job")


def normalize_train_job(job: dict) -> dict:
    raw = _require_mapping(job, label="job")
    _reject_unknown_fields(raw, allowed=JOB_TOP_LEVEL_FIELDS, label="top-level")
    _require_keys(raw, required=JOB_TOP_LEVEL_FIELDS, label="top-level")

    pool = _require_mapping(raw["pool"], label="pool")
    _reject_unknown_fields(pool, allowed=JOB_POOL_FIELDS, label="pool")
    _require_keys(pool, required=JOB_POOL_FIELDS, label="pool")

    model = _require_mapping(raw["model"], label="model")
    _reject_unknown_fields(model, allowed=FULL_MODEL_FIELDS, label="model")
    _require_keys(model, required=FULL_MODEL_FIELDS, label="model")

    train = _require_mapping(raw["train"], label="train")
    _reject_unknown_fields(train, allowed=FULL_TRAIN_FIELDS, label="train")
    _require_keys(train, required=FULL_TRAIN_FIELDS, label="train")

    normalized = {
        "dataset": _require_string(raw["dataset"], field_name="dataset"),
        "pool": {
            "name": _require_string(pool["name"], field_name="pool.name"),
            "ratio": _normalize_float(pool["ratio"], field_name="pool.ratio"),
        },
        "model": {
            "hidden_features": _normalize_int(model["hidden_features"], field_name="model.hidden_features"),
            "nonlinearity": _require_string(model["nonlinearity"], field_name="model.nonlinearity"),
            "p_dropout": _normalize_float(model["p_dropout"], field_name="model.p_dropout"),
            "conv_layer": _require_string(model["conv_layer"], field_name="model.conv_layer"),
            "pre_gnn": [_normalize_int(value, field_name="model.pre_gnn[]") for value in model["pre_gnn"]],
            "post_gnn": [_normalize_int(value, field_name="model.post_gnn[]") for value in model["post_gnn"]],
            "variant": _require_string(model["variant"], field_name="model.variant"),
        },
        "train": {
            "runs": _normalize_int(train["runs"], field_name="train.runs"),
            "lr": _normalize_float(train["lr"], field_name="train.lr"),
            "batch_size": _normalize_int(train["batch_size"], field_name="train.batch_size"),
            "patience": _normalize_int(train["patience"], field_name="train.patience"),
            "epochs": _normalize_int(train["epochs"], field_name="train.epochs"),
            "train_ratio": _normalize_float(train["train_ratio"], field_name="train.train_ratio"),
            "val_ratio": _normalize_float(train["val_ratio"], field_name="train.val_ratio"),
            "seed_mode": _require_string(train["seed_mode"], field_name="train.seed_mode"),
            "seed_base": _normalize_int(train["seed_base"], field_name="train.seed_base"),
            "seed_list": _normalize_seed_list(train["seed_list"]),
            "allow_duplicate_seeds": _normalize_bool(
                train["allow_duplicate_seeds"],
                field_name="train.allow_duplicate_seeds",
            ),
        },
        "log_file": _normalize_optional_string(raw["log_file"], field_name="log_file"),
        "tag": _normalize_optional_string(raw["tag"], field_name="tag"),
    }

    validate_dataset(normalized["dataset"])
    validate_pool_ratio(normalized["pool"]["ratio"])
    validate_pool(normalized["pool"]["name"])
    validate_model_type(normalized["model"]["variant"])

    if normalized["train"]["seed_mode"] not in {"auto", "file", "list"}:
        raise ValueError("train.seed_mode must be 'auto', 'file', or 'list'.")

    train_ratio = normalized["train"]["train_ratio"]
    val_ratio = normalized["train"]["val_ratio"]
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Invalid split ratio. Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1."
        )
    if normalized["train"]["runs"] <= 0:
        raise ValueError("Invalid runs value. Require train.runs > 0.")
    if normalized["train"]["seed_list"] is not None and normalized["train"]["seed_mode"] != "list":
        raise ValueError("train.seed_mode must be 'list' when train.seed_list is provided in a complete job.")

    return normalized


def compute_train_job_case_id(job: dict) -> str:
    payload = {
        "dataset": job["dataset"],
        "pool": {
            "name": job["pool"]["name"],
            "ratio": job["pool"]["ratio"],
        },
        "model": {
            "hidden_features": job["model"]["hidden_features"],
            "nonlinearity": job["model"]["nonlinearity"],
            "p_dropout": job["model"]["p_dropout"],
            "conv_layer": job["model"]["conv_layer"],
            "pre_gnn": job["model"]["pre_gnn"],
            "post_gnn": job["model"]["post_gnn"],
            "variant": job["model"]["variant"],
        },
        "train": {
            "runs": job["train"]["runs"],
            "lr": job["train"]["lr"],
            "batch_size": job["train"]["batch_size"],
            "patience": job["train"]["patience"],
            "epochs": job["train"]["epochs"],
            "train_ratio": job["train"]["train_ratio"],
            "val_ratio": job["train"]["val_ratio"],
            "seed_mode": job["train"]["seed_mode"],
            "seed_base": job["train"]["seed_base"],
            "seed_list": job["train"]["seed_list"],
            "allow_duplicate_seeds": job["train"]["allow_duplicate_seeds"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


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
    _require_mapping(model_conf.get("model"), label="model config")
    _require_mapping(experiment_conf.get("experiment"), label="experiment config")
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
