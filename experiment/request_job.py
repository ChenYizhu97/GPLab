from copy import deepcopy

from .request_common import TrainRequestContext, build_internal_request


def build_job_request(job: dict) -> TrainRequestContext:
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
    conf = build_internal_request(
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
    return TrainRequestContext(
        conf=conf,
        log_file=normalized["log_file"],
        seed_mode=normalized["train"]["seed_mode"],
        seed_base=normalized["train"]["seed_base"],
        allow_duplicate_seeds=normalized["train"]["allow_duplicate_seeds"],
        seed_list=deepcopy(normalized["train"]["seed_list"]),
    )
