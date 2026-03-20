import json
import typer
from typing_extensions import Optional, Annotated
import numpy as np

app = typer.Typer(pretty_exceptions_enable=False)
REQUIRED_REPRO_FIELDS = [
    "seed_mode",
    "seeds",
    "split_digest",
    "split_ratio",
    "dataset_id",
    "env",
]


def _load_jsonstream(file: str):
    with open(file, mode="r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _filter_pool(pool: str, data: list[dict]):
    return filter(lambda x: x.get("pool", {}).get("method") == pool, data)


def _filter_dataset(dataset: str, data: list[dict]):
    return filter(lambda x: x.get("dataset", "").lower() == dataset.lower(), data)


def _filter_comment(comment: str, data: list[dict]):
    return filter(lambda x: x.get("comment", "") == comment, data)


def _read_epochs(record: dict):
    return {"avg_epoch": float(np.mean(record["results"]["data"]["epochs_stop"]))}


def _read_statistic(record: dict):
    return record["results"]["statistic"]


def _read_default(record: dict):
    default = {**record["pool"], "dataset": record["dataset"]}
    if "comment" in record:
        default["comment"] = record["comment"]
    return default


def _read_repro(record: dict):
    repro = record.get("repro")
    if repro is None:
        raise ValueError("Record missing top-level 'repro'.")

    missing = [field for field in REQUIRED_REPRO_FIELDS if field not in repro]
    if missing:
        raise ValueError(f"Record has incomplete repro fields: {missing}")

    return {
        "repro_status": "ok",
        "seed_mode": repro["seed_mode"],
        "seed_base": repro.get("seed_base"),
        "seeds": repro["seeds"],
        "split_digest": repro["split_digest"],
        "split_ratio": repro["split_ratio"],
        "dataset_id": repro["dataset_id"],
        "env": repro["env"],
        "missing_repro_fields": [],
    }


def _read(record: dict, epoch: bool = False, show_repro: bool = False, verify_repro: bool = False):
    result = {**_read_default(record), **_read_statistic(record)}

    if epoch:
        result.update(_read_epochs(record))

    if show_repro or verify_repro:
        result.update(_read_repro(record))

    if verify_repro:
        result["repro_verified"] = True

    return result


@app.command()
def main(
        file: Annotated[str, typer.Argument()],
        pool: Annotated[Optional[str], typer.Option()] = None,
        dataset: Annotated[Optional[str], typer.Option()] = None,
        comment: Annotated[Optional[str], typer.Option()] = None,
        epoch: Annotated[Optional[bool], typer.Option()] = False,
        show_repro: Annotated[Optional[bool], typer.Option(help="Show reproducibility fields.")] = False,
        verify_repro: Annotated[Optional[bool], typer.Option(help="Verify reproducibility field completeness.")] = False,
):
    records = _load_jsonstream(file)
    if dataset is not None:
        records = _filter_dataset(dataset, records)
    if pool is not None:
        records = _filter_pool(pool, records)
    if comment is not None:
        records = _filter_comment(comment, records)

    for record in records:
        print(_read(record, epoch, show_repro, verify_repro))


if __name__ == "__main__":
    app()
