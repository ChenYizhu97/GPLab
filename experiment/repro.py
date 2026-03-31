import hashlib
import json
from typing import Optional

def build_protocol_digest(conf: dict) -> str:
    expr_conf = conf["experiment"]
    payload = {
        "dataset": conf["dataset"],
        "model": conf["model"],
        "pool": conf["pool"],
        "experiment": {
            "runs": expr_conf["runs"],
            "lr": expr_conf["lr"],
            "batch_size": expr_conf["batch_size"],
            "patience": expr_conf["patience"],
            "epochs": expr_conf["epochs"],
            "train_ratio": expr_conf["train_ratio"],
            "val_ratio": expr_conf["val_ratio"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def build_replay_block(
    conf: dict,
    *,
    seed_mode: str,
    seed_base: Optional[int],
) -> dict:
    return {
        "preferred_seed_mode": "list",
        "seed_list": conf["experiment"]["seeds"],
        "source_seed_mode": seed_mode,
        "source_seed_base": seed_base,
        "selection": {
            "dataset": conf["dataset"],
            "pool": conf["pool"]["method"],
            "model_type": conf["model"].get("variant", "sum"),
            "comment": conf.get("comment"),
        },
    }
