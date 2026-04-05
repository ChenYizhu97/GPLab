import shlex

import typer
from typing_extensions import Annotated, Optional

from experiment.identity import compute_benchmark_key, ensure_record_id
from experiment.record import summarize_record
from utils.cli import validate_dataset, validate_model_type, validate_pool
from utils.jsonl import read_jsonl
from utils.presentation import build_error_payload, emit_json, validate_output_format

app = typer.Typer(pretty_exceptions_enable=False)


def _load_jsonl(path: str) -> list[dict]:
    return [ensure_record_id(record) for record in read_jsonl(path)]


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


def _group_header(records: list[dict]) -> str:
    first = records[0]
    spec = first["spec"]
    tags = sorted({record.get("tag") for record in records if record.get("tag") is not None})
    parts = [
        f"dataset={spec['dataset']}",
        f"model={spec['model'].get('variant', 'sum')}",
        f"benchmark={compute_benchmark_key(first)}",
    ]
    if len(tags) == 1:
        parts.append(f"tag={tags[0]}")
    elif len(tags) > 1:
        parts.append(f"tags={len(tags)}")
    return " | ".join(parts)


def _sort_value(record: dict, sort_by: str) -> float:
    summary = summarize_record(record)
    return float(summary[sort_by])


def _print_report(records: list[dict], sort_by: str) -> None:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(compute_benchmark_key(record), []).append(record)

    for group in groups.values():
        ranked = sorted(
            group,
            key=lambda record: _sort_value(record, sort_by),
            reverse=sort_by not in {"std", "avg_val_loss", "avg_best_epoch"},
        )
        print(_group_header(ranked))
        for index, record in enumerate(ranked, start=1):
            summary = summarize_record(record)
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


def _build_report_payload(records: list[dict], sort_by: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(compute_benchmark_key(record), []).append(record)

    payload_groups = []
    for benchmark_key, group in groups.items():
        ranked = sorted(
            group,
            key=lambda record: _sort_value(record, sort_by),
            reverse=sort_by not in {"std", "avg_val_loss", "avg_best_epoch"},
        )
        group_summaries = []
        for index, record in enumerate(ranked, start=1):
            summary = summarize_record(record)
            summary["rank"] = index
            group_summaries.append(summary)

        first = ranked[0]
        payload_groups.append(
            {
                "benchmark_key": benchmark_key,
                "dataset": first["spec"]["dataset"],
                "model_type": first["spec"]["model"].get("variant", "sum"),
                "records": group_summaries,
            }
        )

    return {
        "ok": True,
        "kind": "query_report",
        "groups": payload_groups,
    }


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
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    try:
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
            if output_format == "json":
                emit_json(_build_report_payload(records, sort_by))
                return
            _print_report(records, sort_by)
            return

        summaries = []
        for record in records:
            summary = summarize_record(record)
            if show_spec:
                summary["spec"] = record["spec"]
            if show_replay:
                summary["replay_command"] = (
                    f"python3 replay.py --log-file {shlex.quote(log_file)} --record-id {record['record_id']}"
                )
            summaries.append(summary)

        if output_format == "json":
            emit_json({"ok": True, "kind": "query_result", "records": summaries})
            return

        for summary in summaries:
            print(summary)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("query_error", exc, details={"log_file": log_file}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
