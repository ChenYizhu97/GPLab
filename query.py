import hashlib
import json
import shlex

import numpy as np
import typer
from typing_extensions import Annotated, Optional

from experiment.identity import ensure_record_id
from utils.cli import validate_dataset, validate_model_type, validate_pool
from utils.jsonl import read_jsonl

app = typer.Typer(pretty_exceptions_enable=False)


def _load_jsonl(path: str) -> list[dict]:
    return [ensure_record_id(record) for record in read_jsonl(path)]


def _record_summary(record: dict) -> dict:
    runs = record["result"]["runs"]
    test_acc = [float(run["best_test_acc"]) for run in runs]
    val_loss = [float(run["best_val_loss"]) for run in runs]
    epochs = [int(run["best_epoch"]) for run in runs]

    corr = None
    if len(runs) >= 2 and np.std(val_loss) != 0 and np.std(test_acc) != 0:
        corr = float(np.corrcoef(val_loss, test_acc)[0, 1])

    summary = {
        "record_id": record["record_id"],
        "dataset": record["spec"]["dataset"],
        "pool": record["spec"]["pool"]["name"],
        "pool_ratio": record["spec"]["pool"]["ratio"],
        "model_type": record["spec"]["model"].get("variant", "sum"),
        "runs": len(runs),
        "mean": float(record["result"]["mean"]),
        "std": float(record["result"]["std"]),
        "avg_best_epoch": float(np.mean(epochs)),
        "avg_val_loss": float(np.mean(val_loss)),
        "best_test_acc": float(max(test_acc)),
        "worst_test_acc": float(min(test_acc)),
        "val_loss_test_acc_corr": corr,
    }
    if "tag" in record:
        summary["tag"] = record["tag"]
    return summary


def _matches(record: dict, *, dataset: Optional[str], pool: Optional[str], tag: Optional[str], model_type: Optional[str]) -> bool:
    spec = record["spec"]
    if dataset is not None and spec["dataset"].lower() != dataset.lower():
        return False
    if pool is not None and spec["pool"]["name"] != pool:
        return False
    if tag is not None and record.get("tag") != tag:
        return False
    if model_type is not None and spec["model"].get("variant", "sum") != model_type:
        return False
    return True


def _benchmark_key(record: dict) -> str:
    spec = record["spec"]
    payload = {
        "dataset": spec["dataset"],
        "model": spec["model"],
        "train": spec["train"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _group_header(records: list[dict]) -> str:
    first = records[0]
    spec = first["spec"]
    tags = sorted({record.get("tag") for record in records if record.get("tag") is not None})
    parts = [
        f"dataset={spec['dataset']}",
        f"model={spec['model'].get('variant', 'sum')}",
        f"benchmark={_benchmark_key(first)}",
    ]
    if len(tags) == 1:
        parts.append(f"tag={tags[0]}")
    elif len(tags) > 1:
        parts.append(f"tags={len(tags)}")
    return " | ".join(parts)


def _sort_value(record: dict, sort_by: str) -> float:
    summary = _record_summary(record)
    return float(summary[sort_by])


def _print_report(records: list[dict], sort_by: str) -> None:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(_benchmark_key(record), []).append(record)

    for group in groups.values():
        ranked = sorted(
            group,
            key=lambda record: _sort_value(record, sort_by),
            reverse=sort_by not in {"std", "avg_val_loss", "avg_best_epoch"},
        )
        print(_group_header(ranked))
        for index, record in enumerate(ranked, start=1):
            summary = _record_summary(record)
            corr = summary["val_loss_test_acc_corr"]
            corr_text = "n/a" if corr is None else f"{corr:.4f}"
            print(
                f"{index}. pool={summary['pool']} ratio={summary['pool_ratio']} "
                f"mean={summary['mean']:.4f} std={summary['std']:.4f} "
                f"avg_epoch={summary['avg_best_epoch']:.1f} avg_val_loss={summary['avg_val_loss']:.6f} "
                f"val_test_corr={corr_text} record_id={summary['record_id']}"
            )
        print(
            "Interpretation: compare mean first, then use std for stability, avg_epoch for early-stop behavior, "
            "and val_test_corr to judge whether lower validation loss really aligned with better test accuracy."
        )
        print()


@app.command()
def main(
    log_file: Annotated[str, typer.Option(..., help="JSONL log file to query.")],
    pool: Annotated[Optional[str], typer.Option()] = None,
    dataset: Annotated[Optional[str], typer.Option()] = None,
    model_type: Annotated[Optional[str], typer.Option(help="Filter by model variant: sum or plain.")] = None,
    tag: Annotated[Optional[str], typer.Option(help="Filter by experiment tag.")] = None,
    report: Annotated[bool, typer.Option(help="Print grouped benchmark report instead of one summary dict per record.")] = False,
    sort_by: Annotated[str, typer.Option(help="Report sort field: mean, std, avg_best_epoch, avg_val_loss.")] = "mean",
    show_spec: Annotated[bool, typer.Option(help="Include the full spec block in default output.")] = False,
    show_replay: Annotated[bool, typer.Option(help="Show replay.py command for each matched record.")] = False,
):
    if sort_by not in {"mean", "std", "avg_best_epoch", "avg_val_loss"}:
        raise typer.BadParameter(
            "sort_by must be one of: mean, std, avg_best_epoch, avg_val_loss.",
            param_hint="--sort-by",
        )

    if dataset is not None:
        validate_dataset(dataset)
    if pool is not None:
        validate_pool(pool)
    if model_type is not None:
        validate_model_type(model_type)

    records = [
        record
        for record in _load_jsonl(log_file)
        if _matches(record, dataset=dataset, pool=pool, tag=tag, model_type=model_type)
    ]

    if report:
        _print_report(records, sort_by)
        return

    for record in records:
        summary = _record_summary(record)
        if show_spec:
            summary["spec"] = record["spec"]
        if show_replay:
            summary["replay_command"] = (
                f"python3 replay.py --log-file {shlex.quote(log_file)} --record-id {record['record_id']}"
            )
        print(summary)


if __name__ == "__main__":
    app()
