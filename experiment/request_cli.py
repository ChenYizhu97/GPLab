from copy import deepcopy
from typing import Optional

from .request_common import build_internal_request, resolve_request_seed_options


def build_cli_request(
    *,
    model_conf: dict,
    experiment_conf: dict,
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

    exp = merged_experiment_conf.get("experiment", {})
    final_seed_mode, final_seed_base, final_allow_dup, final_seed_list = resolve_request_seed_options(
        seed_mode=seed_mode,
        seed_base=seed_base,
        seed_list=seed_list,
        allow_duplicate_seeds=allow_duplicate_seeds,
        expr_conf=exp,
    )

    conf = build_internal_request(
        model_conf=merged_model_conf,
        experiment_conf=merged_experiment_conf,
        pool=pool if pool is not None else "nopool",
        pool_ratio=float(pool_ratio if pool_ratio is not None else 0.5),
        dataset_name=dataset_name if dataset_name is not None else "PROTEINS",
        model_type=model_type if model_type is not None else "sum",
        tag=tag,
        seed_mode=final_seed_mode,
        seed_base=final_seed_base,
        seed_list=final_seed_list,
        allow_duplicate_seeds=final_allow_dup,
    )
    return conf, log_file, final_seed_mode, final_seed_base, final_allow_dup, final_seed_list
