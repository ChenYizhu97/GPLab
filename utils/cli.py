import typer
from typing import Optional
from utils.registry import TU_DATASETS, BUILTIN_POOLS


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
    if ratio <= 0.0 or ratio > 1.0:
        raise typer.BadParameter("pool_ratio must be in (0, 1].", param_hint="--pool-ratio")


def validate_model_type(value: str) -> None:
    if value not in {"sum", "plain"}:
        raise typer.BadParameter("model_type must be 'sum' or 'plain'.", param_hint="--model-type")


def normalize_seed_mode(mode: str) -> str:
    if mode not in {"auto", "file"}:
        raise typer.BadParameter("seed_mode must be 'auto' or 'file'.", param_hint="--seed-mode")
    return mode


def resolve_seed_options(
    seed_mode: Optional[str],
    seed_base: Optional[int],
    allow_duplicate_seeds: Optional[bool],
    expr_conf: dict,
) -> tuple[str, int, bool]:
    conf = expr_conf or {}

    final_seed_mode = seed_mode if seed_mode is not None else conf.get("seed_mode", "auto")
    final_seed_mode = normalize_seed_mode(final_seed_mode)

    final_seed_base = seed_base if seed_base is not None else conf.get("seed_base", 20260320)
    if final_seed_base is None:
        final_seed_base = 20260320

    final_allow_dup = allow_duplicate_seeds
    if final_allow_dup is None:
        final_allow_dup = conf.get("allow_duplicate_seeds", False)
    final_allow_dup = bool(final_allow_dup)

    return final_seed_mode, int(final_seed_base), final_allow_dup
