from copy import deepcopy
from typing import Optional

import numpy as np

from experiment.identity import compute_benchmark_key, ensure_record_id
from utils.jsonl import append_jsonl


def build_spec(conf: dict) -> dict:
    expr_conf = conf["experiment"]
    split = {
        "train": float(expr_conf["train_ratio"]),
        "val": float(expr_conf["val_ratio"]),
        "test": float(1.0 - expr_conf["train_ratio"] - expr_conf["val_ratio"]),
    }
    spec = {
        "dataset": conf["dataset"],
        "model": deepcopy(conf["model"]),
        "pool": {
            "name": conf["pool"]["method"],
            "ratio": conf["pool"]["ratio"],
            "source": conf["pool"].get("source", "builtin"),
        },
        "train": {
            "lr": float(expr_conf["lr"]),
            "batch_size": int(expr_conf["batch_size"]),
            "patience": int(expr_conf["patience"]),
            "epochs": int(expr_conf["epochs"]),
            "split": split,
            "seeds": [int(seed) for seed in expr_conf["seeds"]],
        },
    }
    return spec


def build_result(run_records: list[dict]) -> dict:
    if not run_records:
        raise ValueError("Cannot build result from an empty run record list.")

    test_acc = [float(run["best_test_acc"]) for run in run_records]
    compact_runs = [
        {
            "seed": int(run["seed"]),
            "best_epoch": int(run["best_epoch"]),
            "best_val_loss": float(run["best_val_loss"]),
            "best_test_acc": float(run["best_test_acc"]),
        }
        for run in run_records
    ]
    return {
        "mean": float(np.mean(test_acc)),
        "std": float(np.std(test_acc)),
        "runs": compact_runs,
    }


def build_record(conf: dict, *, runtime: dict, run_records: list[dict]) -> dict:
    record = {
        "spec": build_spec(conf),
        "runtime": runtime,
        "result": build_result(run_records),
    }
    if conf.get("tag") is not None:
        record["tag"] = conf["tag"]
    ensure_record_id(record)
    return record


def append_record_if_needed(log_file: Optional[str], record: dict) -> None:
    if log_file is not None:
        append_jsonl(log_file, record)


def summarize_record(record: dict) -> dict:
    ensured = ensure_record_id(record)
    runs = ensured["result"]["runs"]
    test_acc = [float(run["best_test_acc"]) for run in runs]
    val_loss = [float(run["best_val_loss"]) for run in runs]
    epochs = [int(run["best_epoch"]) for run in runs]

    corr = None
    if len(runs) >= 2 and np.std(val_loss) != 0 and np.std(test_acc) != 0:
        corr = float(np.corrcoef(val_loss, test_acc)[0, 1])

    summary = {
        "record_id": ensured["record_id"],
        "benchmark_key": compute_benchmark_key(ensured),
        "dataset": ensured["spec"]["dataset"],
        "pool": ensured["spec"]["pool"]["name"],
        "pool_ratio": ensured["spec"]["pool"]["ratio"],
        "model_type": ensured["spec"]["model"].get("variant", "sum"),
        "runs": len(runs),
        "mean": float(ensured["result"]["mean"]),
        "std": float(ensured["result"]["std"]),
        "avg_best_epoch": float(np.mean(epochs)),
        "avg_val_loss": float(np.mean(val_loss)),
        "best_test_acc": float(max(test_acc)),
        "worst_test_acc": float(min(test_acc)),
        "val_loss_test_acc_corr": corr,
    }
    if "tag" in ensured:
        summary["tag"] = ensured["tag"]
    return summary
