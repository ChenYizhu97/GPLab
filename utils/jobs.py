import json
from pathlib import Path


JOB_TOP_LEVEL_FIELDS = {"dataset", "pool", "model", "train", "log_file", "tag"}
JOB_POOL_FIELDS = {"name", "ratio"}
JOB_MODEL_FIELDS = {"variant"}
JOB_TRAIN_FIELDS = {
    "runs",
    "lr",
    "batch_size",
    "patience",
    "epochs",
    "train_ratio",
    "val_ratio",
    "seed_mode",
    "seed_base",
    "seed_list",
    "allow_duplicate_seeds",
}


def _require_mapping(value, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _reject_unknown_fields(payload: dict, *, allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {label} field(s): {joined}.")


def load_job_file(path: str) -> dict:
    job_path = Path(path)
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found: {path}")

    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid job JSON: {exc}") from exc

    job = _require_mapping(payload, label="job")
    _reject_unknown_fields(job, allowed=JOB_TOP_LEVEL_FIELDS, label="top-level")

    if "pool" in job:
        pool = _require_mapping(job["pool"], label="pool")
        _reject_unknown_fields(pool, allowed=JOB_POOL_FIELDS, label="pool")

    if "model" in job:
        model = _require_mapping(job["model"], label="model")
        _reject_unknown_fields(model, allowed=JOB_MODEL_FIELDS, label="model")

    if "train" in job:
        train = _require_mapping(job["train"], label="train")
        _reject_unknown_fields(train, allowed=JOB_TRAIN_FIELDS, label="train")
        if "seed_list" in train:
            seed_list = train["seed_list"]
            if not isinstance(seed_list, list) or not seed_list:
                raise ValueError("train.seed_list must be a non-empty array of integers.")
            train["seed_list"] = [int(value) for value in seed_list]

    return job
