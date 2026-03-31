from typing import Optional

import numpy as np

from utils.jsonl import append_jsonl


def build_repro_block(conf: dict, seed_mode: str, seed_base: Optional[int], dataset_id: dict) -> dict:
    return {
        "seed_mode": seed_mode,
        "seed_base": seed_base,
        "seeds": conf["experiment"]["seeds"],
        "split_digest": conf["experiment"]["split_digest"],
        "split_ratio": conf["experiment"]["split_ratio"],
        "dataset_id": dataset_id,
        "env": conf["meta"],
    }


def attach_runtime_meta(conf: dict, meta: dict) -> dict:
    conf["meta"] = meta
    return conf


def attach_results(conf: dict, *, val_loss: list[float], test_acc: list[float], epochs_stop: list[int], runs: list[dict]) -> dict:
    if not test_acc:
        raise ValueError("Cannot attach results from an empty experiment run list.")

    conf["results"] = {
        "statistic": {
            "mean": float(np.mean(test_acc)),
            "std": float(np.std(test_acc)),
        },
        "data": {
            "val_loss": val_loss,
            "test_acc": test_acc,
            "epochs_stop": epochs_stop,
            "runs": runs,
        },
    }
    return conf


def attach_repro(conf: dict, *, seed_mode: str, seed_base: Optional[int], dataset_id: dict) -> dict:
    conf["repro"] = build_repro_block(conf, seed_mode=seed_mode, seed_base=seed_base, dataset_id=dataset_id)
    return conf


def append_record_if_needed(log_file: Optional[str], record: dict) -> None:
    if log_file is not None:
        append_jsonl(log_file, record)
