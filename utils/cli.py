import math
from typing import Optional

import typer

from utils.registry import TU_DATASETS, BUILTIN_POOLS
from utils.presentation import validate_output_format


def validate_dataset(name: str) -> None:
    if name not in TU_DATASETS:
        raise typer.BadParameter(
            f"Unsupported dataset '{name}'. Supported datasets: {', '.join(TU_DATASETS)}",
            param_hint="--dataset",
        )


def validate_pool(name: str, builtins: tuple[str, ...] = BUILTIN_POOLS) -> bool:
    is_custom_pool = ":" in name
    if not is_custom_pool and name not in builtins:
        raise typer.BadParameter(
            f"Unknown pooling method '{name}'. Built-ins: {', '.join(builtins)}",
            param_hint="--pool",
        )
    return is_custom_pool


def validate_pool_ratio(ratio: float) -> None:
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise typer.BadParameter("pool_ratio must be in (0, 1].", param_hint="--pool-ratio")


def validate_model_type(value: str) -> None:
    if value not in {"sum", "plain"}:
        raise typer.BadParameter("model_type must be 'sum' or 'plain'.", param_hint="--model-type")


def normalize_seed_mode(mode: str) -> str:
    if mode not in {"auto", "file", "list"}:
        raise typer.BadParameter("seed_mode must be 'auto', 'file', or 'list'.", param_hint="--seed-mode")
    return mode


def parse_seed_list(seed_list: Optional[str]) -> Optional[list[int]]:
    if seed_list is None:
        return None

    raw_tokens = [token.strip() for token in seed_list.split(",")]
    values = [token for token in raw_tokens if token]
    if not values:
        raise typer.BadParameter("seed_list must contain at least one integer.", param_hint="--seed-list")

    try:
        return [int(token) for token in values]
    except ValueError as exc:
        raise typer.BadParameter(
            "seed_list must be a comma-separated list of integers.",
            param_hint="--seed-list",
        ) from exc


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    values = [item for item in items if item]
    if not values:
        raise typer.BadParameter("Expected at least one non-empty comma-separated value.")
    return values


def _normalize_config_seed(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise typer.BadParameter(
            "experiment.seed_list in config must be a list of integers.",
            param_hint="--seed-list",
        )
    return int(value)


def resolve_seed_options(
    seed_mode: Optional[str],
    seed_base: Optional[int],
    seed_list: Optional[str],
    allow_duplicate_seeds: Optional[bool],
    expr_conf: dict,
) -> tuple[str, int, bool, Optional[list[int]]]:
    conf = expr_conf or {}
    final_seed_list = parse_seed_list(seed_list)

    if final_seed_list is None:
        conf_seed_list = conf.get("seed_list")
        if conf_seed_list is not None:
            if not isinstance(conf_seed_list, list):
                raise typer.BadParameter(
                    "experiment.seed_list in config must be a list of integers.",
                    param_hint="--seed-list",
                )
            final_seed_list = [_normalize_config_seed(value) for value in conf_seed_list]

    final_seed_mode = seed_mode if seed_mode is not None else conf.get("seed_mode", "auto")
    if final_seed_list is not None:
        final_seed_mode = "list"
    final_seed_mode = normalize_seed_mode(final_seed_mode)

    final_seed_base = seed_base if seed_base is not None else conf.get("seed_base", 20260320)
    if final_seed_base is None:
        final_seed_base = 20260320

    final_allow_dup = allow_duplicate_seeds
    if final_allow_dup is None:
        final_allow_dup = conf.get("allow_duplicate_seeds", False)
    final_allow_dup = bool(final_allow_dup)

    return final_seed_mode, int(final_seed_base), final_allow_dup, final_seed_list
