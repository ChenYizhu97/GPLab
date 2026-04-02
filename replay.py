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
from utils.jsonl import read_jsonl
from utils.io import build_runtime_meta

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


def _build_command(*, model_path: Path, experiment_path: Path, record: dict, replay_log_file: Optional[str]) -> list[str]:
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


@app.command()
def main(
    log_file: Annotated[str, typer.Option(..., help="JSONL log file containing the record to replay.")],
    record_id: Annotated[str, typer.Option(help="Record id of the JSONL entry to replay.")] = ...,
    output_dir: Annotated[str, typer.Option(help="Directory for generated replay configs.")] = "/tmp/gplab_replay",
    replay_log_file: Annotated[Optional[str], typer.Option(help="Optional JSONL file to append the replayed result to.")] = None,
    run: Annotated[bool, typer.Option(help="Execute the replay command after generating configs.")] = False,
):
    records = [ensure_record_id(record) for record in read_jsonl(log_file)]
    record = _pick_record(records, record_id=record_id)
    target_dir = _replay_dir(output_dir, record)
    model_path, experiment_path = _materialize_configs(target_dir, record)
    command = _build_command(
        model_path=model_path,
        experiment_path=experiment_path,
        record=record,
        replay_log_file=replay_log_file,
    )

    print(f"Replay directory: {target_dir}")
    print(f"Replay command: {_stringify_command(command)}")

    current_runtime = _current_runtime()
    mismatches = _compare_runtime(record.get("runtime", {}), current_runtime)
    if mismatches:
        print("Runtime compatibility warning:")
        for item in mismatches:
            print(f"  - {item}")
    else:
        print("Runtime compatibility: current environment matches recorded runtime on checked fields.")

    if run:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    app()
