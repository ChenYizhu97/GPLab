import shlex
import typer
from typing_extensions import Optional, Annotated
import numpy as np
from experiment.identity import ensure_record_id
from utils.cli import validate_dataset, validate_model_type, validate_pool
from utils.jsonl import read_jsonl

app = typer.Typer(pretty_exceptions_enable=False)
REQUIRED_REPRO_FIELDS = [
    "protocol_digest",
    "seed_mode",
    "seeds",
    "split_digest",
    "split_ratio",
    "dataset_id",
    "env",
    "replay",
]


def _load_jsonl(path: str) -> list[dict]:
    return read_jsonl(path)


def _filter_pool(pool: str, data):
    return filter(lambda record: record.get("pool", {}).get("method") == pool, data)


def _filter_dataset(dataset: str, data):
    return filter(lambda record: record.get("dataset", "").lower() == dataset.lower(), data)


def _filter_comment(comment: str, data):
    return filter(lambda record: record.get("comment", "") == comment, data)


def _filter_model_type(model_type: str, data):
    return filter(lambda record: record.get("model", {}).get("variant", "sum") == model_type, data)


def _apply_filters(
    records,
    dataset: Optional[str],
    pool: Optional[str],
    comment: Optional[str],
    model_type: Optional[str],
):
    filtered = records
    if dataset is not None:
        filtered = _filter_dataset(dataset, filtered)
    if pool is not None:
        filtered = _filter_pool(pool, filtered)
    if comment is not None:
        filtered = _filter_comment(comment, filtered)
    if model_type is not None:
        filtered = _filter_model_type(model_type, filtered)
    return filtered


def _read_epochs(record: dict):
    return {"avg_best_epoch": float(np.mean(record["results"]["data"]["epochs_stop"]))}


def _read_statistic(record: dict):
    return record["results"]["statistic"]


def _read_default(record: dict):
    default = {**record["pool"], "dataset": record["dataset"]}
    default["model_type"] = record.get("model", {}).get("variant", "sum")
    if "comment" in record:
        default["comment"] = record["comment"]
    return default


def _extract_repro(record: dict) -> tuple[Optional[dict], list[str]]:
    repro = record.get("repro")
    if repro is None:
        return None, list(REQUIRED_REPRO_FIELDS)
    missing = [field for field in REQUIRED_REPRO_FIELDS if field not in repro]
    return repro, missing


def _shape_output(
    record: dict,
    *,
    log_file: str,
    epoch: bool = False,
    show_repro: bool = False,
    verify_repro: bool = False,
    show_replay: bool = False,
):
    result = {**_read_default(record), **_read_statistic(record)}
    result["record_id"] = record["record_id"]

    if epoch:
        result.update(_read_epochs(record))

    if show_repro:
        repro, missing = _extract_repro(record)
        if repro is None:
            raise ValueError("Record missing top-level 'repro'.")
        if missing:
            raise ValueError(f"Record has incomplete repro fields: {missing}")
        result["repro"] = repro

    if verify_repro:
        _, missing = _extract_repro(record)
        result["repro_status"] = "ok" if not missing else "missing"
        result["missing_repro_fields"] = missing

    if show_replay:
        result["replay_command"] = (
            f"python3 replay.py --log-file {shlex.quote(log_file)} --record-id {record['record_id']}"
        )

    return result


@app.command()
def main(
        log_file: Annotated[str, typer.Option(..., help="JSONL log file to query.")],
        pool: Annotated[Optional[str], typer.Option()] = None,
        dataset: Annotated[Optional[str], typer.Option()] = None,
        model_type: Annotated[Optional[str], typer.Option(help="Filter by model variant: sum or plain.")] = None,
        comment: Annotated[Optional[str], typer.Option()] = None,
        epoch: Annotated[bool, typer.Option()] = False,
        show_repro: Annotated[bool, typer.Option(help="Show reproducibility fields.")] = False,
        verify_repro: Annotated[bool, typer.Option(help="Verify reproducibility field completeness.")] = False,
        show_replay: Annotated[bool, typer.Option(help="Show replay.py command for each matched record.")] = False,
):
    if sum((show_repro, verify_repro, show_replay)) > 1:
        raise typer.BadParameter(
            "Use at most one of --show-repro, --verify-repro, or --show-replay.",
            param_hint="--show-repro/--verify-repro/--show-replay",
        )

    if dataset is not None:
        validate_dataset(dataset)
    if pool is not None:
        validate_pool(pool)
    if model_type is not None:
        validate_model_type(model_type)

    records = [ensure_record_id(record) for record in _load_jsonl(log_file)]
    records = _apply_filters(records, dataset, pool, comment, model_type)

    for record in records:
        print(
            _shape_output(
                record,
                log_file=log_file,
                epoch=epoch,
                show_repro=show_repro,
                verify_repro=verify_repro,
                show_replay=show_replay,
            )
        )


if __name__ == "__main__":
    app()
