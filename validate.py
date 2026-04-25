import json
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import toml
import typer
from typing_extensions import Annotated, Optional

from utils.cli import parse_csv_list, parse_seed_list
from utils.jobs import build_execution_plan_from_configs
from utils.presentation import build_error_payload, emit_json, validate_output_format

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    pools: Annotated[str, typer.Option(help="Comma-separated pools to validate.")] = "sagpool,diffpool",
    datasets: Annotated[str, typer.Option(help="Comma-separated datasets to validate.")] = "MUTAG,PROTEINS",
    model_type: Annotated[str, typer.Option(help="Model variant: sum or plain.")] = "sum",
    pool_ratio: Annotated[float, typer.Option(help="Pooling ratio for all cases.")] = 0.5,
    runs: Annotated[int, typer.Option(help="Runs per smoke case.")] = 1,
    epochs: Annotated[int, typer.Option(help="Epochs per smoke case.")] = 1,
    patience: Annotated[int, typer.Option(help="Patience per smoke case.")] = 0,
    lr: Annotated[float, typer.Option(help="Learning rate per smoke case.")] = 0.0005,
    batch_size: Annotated[int, typer.Option(help="Batch size per smoke case.")] = 16,
    train_ratio: Annotated[float, typer.Option(help="Train split ratio.")] = 0.8,
    val_ratio: Annotated[float, typer.Option(help="Validation split ratio.")] = 0.1,
    log_file: Annotated[Optional[str], typer.Option(help="Optional JSONL file to append smoke records.")] = None,
    model_config: Annotated[str, typer.Option(help="Model config path.")] = "config/model.toml",
    experiment_config: Annotated[str, typer.Option(help="Experiment config path.")] = "config/experiment.toml",
    seed_mode: Annotated[Optional[str], typer.Option(help="Seed source mode override.")] = "auto",
    seed_base: Annotated[Optional[int], typer.Option(help="Seed base override.")] = 20260320,
    seed_list: Annotated[Optional[str], typer.Option(help="Comma-separated seed list override.")] = None,
    allow_duplicate_seeds: Annotated[Optional[bool], typer.Option(help="Allow duplicate seeds.")] = False,
    tag_prefix: Annotated[str, typer.Option(help="Tag prefix for smoke runs.")] = "smoke",
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    try:
        model_conf = toml.load(model_config)
        experiment_conf = toml.load(experiment_config)
        experiment_conf.setdefault("experiment", {})
        experiment_conf["experiment"].update(
            {
                "runs": runs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "epochs": epochs,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
            }
        )
        parsed_seed_list = parse_seed_list(seed_list)
        planned_cases = build_execution_plan_from_configs(
            model_conf=model_conf,
            experiment_conf=experiment_conf,
            pools=parse_csv_list(pools),
            datasets=parse_csv_list(datasets),
            model_types=[model_type],
            pool_ratio=pool_ratio,
            tag_prefix=tag_prefix,
            log_file=log_file,
            seed_mode=seed_mode,
            seed_base=seed_base,
            seed_list=parsed_seed_list,
            allow_duplicate_seeds=allow_duplicate_seeds,
        )
        cases = []
        with TemporaryDirectory(prefix="gplab_validate_") as temp_dir:
            for planned_case in planned_cases:
                pool = planned_case["pool"]
                dataset = planned_case["dataset"]
                case_id = planned_case["case_id"]
                command = []
                start = time.perf_counter()
                job_file = Path(temp_dir) / f"{case_id}.json"
                job_file.write_text(json.dumps(planned_case["job"], ensure_ascii=False), encoding="utf-8")
                try:
                    command = [
                        sys.executable,
                        "run_train_job.py",
                        "--job-file",
                        str(job_file),
                        "--output-format",
                        "json",
                    ]
                    completed = subprocess.run(
                        command,
                        check=False,
                        cwd=Path(__file__).resolve().parent,
                        capture_output=True,
                        text=True,
                    )

                    payload = None
                    stdout_text = completed.stdout.strip()
                    if stdout_text:
                        try:
                            payload = json.loads(stdout_text)
                        except json.JSONDecodeError:
                            payload = None

                    if payload is not None and payload.get("ok"):
                        summary = payload["summary"]
                        case = {
                            "case_id": case_id,
                            "pool": pool,
                            "dataset": dataset,
                            "model_type": model_type,
                            "status": "ok",
                            "seconds": round(time.perf_counter() - start, 4),
                            "record_id": summary["record_id"],
                            "command": command,
                            "subprocess": {
                                "returncode": completed.returncode,
                                "job_file": str(job_file),
                            },
                        }
                    else:
                        error = (payload or {}).get("error", {})
                        case = {
                            "case_id": case_id,
                            "pool": pool,
                            "dataset": dataset,
                            "model_type": model_type,
                            "status": "failed",
                            "seconds": round(time.perf_counter() - start, 4),
                            "error_type": error.get("type", "runtime_error"),
                            "message": error.get(
                                "message",
                                completed.stderr.strip() or stdout_text or "subprocess failed",
                            ),
                            "command": command,
                            "subprocess": {
                                "returncode": completed.returncode,
                                "job_file": str(job_file),
                            },
                        }
                except Exception as exc:
                    case = {
                        "case_id": case_id,
                        "pool": pool,
                        "dataset": dataset,
                        "model_type": model_type,
                        "status": "failed",
                        "seconds": round(time.perf_counter() - start, 4),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "command": command,
                        "subprocess": {
                            "job_file": str(job_file),
                        },
                    }
                cases.append(case)

        summary = {
            "total": len(cases),
            "passed": sum(1 for case in cases if case["status"] == "ok"),
            "failed": sum(1 for case in cases if case["status"] == "failed"),
        }
        payload = {
            "ok": summary["failed"] == 0,
            "kind": "validation_result",
            "mode": "smoke",
            "plan": planned_cases,
            "cases": cases,
            "summary": summary,
        }

        if output_format == "json":
            emit_json(payload)
            if summary["failed"] != 0:
                raise typer.Exit(code=1)
            return

        for case in cases:
            if case["status"] == "ok":
                print(f"[{case['pool']}][{case['dataset']}] ok ({case['seconds']}s) record_id={case['record_id']}")
            else:
                print(f"[{case['pool']}][{case['dataset']}] failed ({case['seconds']}s) {case['message']}")
        print(summary)
        if summary["failed"] != 0:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("validation_error", exc))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
