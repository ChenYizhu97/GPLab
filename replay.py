import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import toml
import torch
import typer
from typing_extensions import Annotated

from experiment.identity import ensure_record_id
from experiment.record import summarize_record
from utils.jsonl import read_jsonl
from utils.io import build_runtime_meta
from utils.presentation import build_error_payload, emit_json, validate_output_format

app = typer.Typer(pretty_exceptions_enable=False)


def _pick_record(records: list[dict], *, record_id: str) -> dict:
    for record in records:
        if record["record_id"] == record_id:
            return record
    raise typer.BadParameter(
        f"record_id '{record_id}' was not found in the selected log file.",
        param_hint="--record-id",
    )


def _replay_dir(base_dir: str, record: dict) -> Path:
    spec = record["spec"]
    return Path(base_dir) / f"{record['record_id']}_{spec['dataset']}_{spec['pool']['name']}"


def _materialize_configs(target_dir: Path, record: dict) -> tuple[Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "model.toml"
    experiment_path = target_dir / "experiment.toml"

    spec = record["spec"]
    model_path.write_text(toml.dumps({"model": spec["model"]}), encoding="utf-8")

    split = spec["train"]["split"]
    experiment_payload = {
        "experiment": {
            "runs": len(spec["train"]["seeds"]),
            "lr": spec["train"]["lr"],
            "batch_size": spec["train"]["batch_size"],
            "patience": spec["train"]["patience"],
            "epochs": spec["train"]["epochs"],
            "train_ratio": split["train"],
            "val_ratio": split["val"],
        }
    }
    experiment_path.write_text(toml.dumps(experiment_payload), encoding="utf-8")
    return model_path, experiment_path


def _build_command(
    *,
    model_path: Path,
    experiment_path: Path,
    record: dict,
    replay_log_file: Optional[str],
    output_format: Optional[str] = None,
) -> list[str]:
    spec = record["spec"]
    command = [
        sys.executable,
        "main.py",
        "--pool",
        spec["pool"]["name"],
        "--pool-ratio",
        str(spec["pool"]["ratio"]),
        "--dataset",
        spec["dataset"],
        "--model-type",
        spec["model"].get("variant", "sum"),
        "--model-config",
        str(model_path),
        "--experiment-config",
        str(experiment_path),
        "--seed-list",
        ",".join(str(seed) for seed in spec["train"]["seeds"]),
    ]
    if record.get("tag") is not None:
        command.extend(["--tag", record["tag"]])
    if replay_log_file is not None:
        command.extend(["--log-file", replay_log_file])
    if output_format is not None:
        command.extend(["--output-format", output_format])
    return command


def _stringify_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _current_runtime() -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return build_runtime_meta(device)


def _compare_runtime(recorded: dict, current: dict) -> list[str]:
    checks = [
        ("python_version", "python"),
        ("torch_version", "torch"),
        ("torch_geometric_version", "torch_geometric"),
        ("device", "device"),
        ("cuda_available", "cuda_available"),
    ]
    mismatches = []
    for key, label in checks:
        if recorded.get(key) != current.get(key):
            mismatches.append(f"{label}: recorded={recorded.get(key)!r}, current={current.get(key)!r}")
    return mismatches


def _compatibility_status(recorded: dict, current: dict) -> tuple[str, list[dict]]:
    checks = [
        ("python_version", "python"),
        ("torch_version", "torch"),
        ("torch_geometric_version", "torch_geometric"),
        ("device", "device"),
        ("cuda_available", "cuda_available"),
    ]
    details = []
    missing_required = False
    mismatch_found = False
    for key, label in checks:
        recorded_value = recorded.get(key)
        current_value = current.get(key)
        match = recorded_value == current_value and recorded_value is not None
        if recorded_value is None:
            missing_required = True
        elif recorded_value != current_value:
            mismatch_found = True
        details.append(
            {
                "field": label,
                "recorded": recorded_value,
                "current": current_value,
                "match": match,
            }
        )

    if missing_required:
        return "unknown", details
    if mismatch_found:
        return "mismatch", details
    return "compatible", details


@app.command()
def main(
    log_file: Annotated[str, typer.Option(..., help="JSONL log file containing the record to replay.")],
    record_id: Annotated[str, typer.Option(help="Record id of the JSONL entry to replay.")] = ...,
    output_dir: Annotated[str, typer.Option(help="Directory for generated replay configs.")] = "/tmp/gplab_replay",
    replay_log_file: Annotated[Optional[str], typer.Option(help="Optional JSONL file to append the replayed result to.")] = None,
    run: Annotated[bool, typer.Option(help="Execute the replay command after generating configs.")] = False,
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "text",
):
    output_format = validate_output_format(output_format)
    try:
        records = [ensure_record_id(record) for record in read_jsonl(log_file)]
        record = _pick_record(records, record_id=record_id)
        target_dir = _replay_dir(output_dir, record)
        model_path, experiment_path = _materialize_configs(target_dir, record)
        command = _build_command(
            model_path=model_path,
            experiment_path=experiment_path,
            record=record,
            replay_log_file=replay_log_file,
            output_format="json" if output_format == "json" else None,
        )

        current_runtime = _current_runtime()
        status, details = _compatibility_status(record.get("runtime", {}), current_runtime)
        replay_payload = {
            "ok": True,
            "kind": "replay_result",
            "record": summarize_record(record),
            "paths": {
                "replay_dir": str(target_dir),
                "model_config": str(model_path),
                "experiment_config": str(experiment_path),
                "replay_log_file": replay_log_file,
            },
            "command": {
                "argv": command,
                "shell": _stringify_command(command),
            },
            "compatibility": {
                "status": status,
                "details": details,
            },
        }

        if output_format == "json":
            if run:
                completed = subprocess.run(
                    command,
                    check=False,
                    cwd=Path(__file__).resolve().parent,
                    capture_output=True,
                    text=True,
                )
                replay_payload["rerun"] = {
                    "requested": True,
                    "ok": completed.returncode == 0,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "appended_to_log": replay_log_file is not None,
                }
                try:
                    replay_payload["rerun"]["payload"] = json.loads(completed.stdout) if completed.stdout.strip() else None
                except json.JSONDecodeError:
                    replay_payload["rerun"]["payload"] = None
                rerun_payload = replay_payload["rerun"]["payload"]
                if isinstance(rerun_payload, dict):
                    replay_payload["rerun"]["record_id"] = rerun_payload.get("summary", {}).get("record_id")
                    replay_payload["rerun"]["summary"] = rerun_payload.get("summary")
                replay_payload["ok"] = completed.returncode == 0
                if completed.returncode != 0:
                    replay_payload["kind"] = "replay_error"
                    emit_json(replay_payload)
                    raise typer.Exit(code=1)
            emit_json(replay_payload)
            return

        print(f"Replay directory: {target_dir}")
        print(f"Replay command: {_stringify_command(command)}")
        if status == "compatible":
            print("Runtime compatibility: current environment matches recorded runtime on checked fields.")
        else:
            print(f"Runtime compatibility: {status}")
            for item in _compare_runtime(record.get('runtime', {}), current_runtime):
                print(f"  - {item}")

        if run:
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("replay_error", exc, details={"log_file": log_file, "record_id": record_id}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
