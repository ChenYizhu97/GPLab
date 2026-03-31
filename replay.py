import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import toml
import typer
from typing_extensions import Annotated

from experiment.identity import ensure_record_id
from utils.jsonl import read_jsonl

app = typer.Typer(pretty_exceptions_enable=False)


def _drop_none(value):
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _pick_record(
    records: list[dict],
    *,
    record_id: str,
) -> dict:
    for record in records:
        if record["record_id"] == record_id:
            return record
    raise typer.BadParameter(
        f"record_id '{record_id}' was not found in the selected log file.",
        param_hint="--record-id",
    )


def _replay_dir(base_dir: str, record: dict) -> Path:
    protocol_digest = record.get("repro", {}).get("protocol_digest", "unknown")
    pool_name = record.get("pool", {}).get("method", "unknown")
    dataset_name = record.get("dataset", "unknown")
    return Path(base_dir) / f"{record['record_id']}_{dataset_name}_{pool_name}_{protocol_digest[:8]}"


def _materialize_configs(target_dir: Path, record: dict) -> tuple[Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "model.toml"
    experiment_path = target_dir / "experiment.toml"

    experiment = record["experiment"]
    experiment_keys = [
        "runs",
        "lr",
        "batch_size",
        "patience",
        "epochs",
        "seeds",
        "seed_mode",
        "seed_base",
        "seed_list",
        "allow_duplicate_seeds",
        "train_ratio",
        "val_ratio",
    ]
    model_payload = {"model": _drop_none(record["model"])}
    experiment_payload = {
        "experiment": _drop_none({key: experiment[key] for key in experiment_keys if key in experiment})
    }

    model_path.write_text(toml.dumps(model_payload), encoding="utf-8")
    experiment_path.write_text(toml.dumps(experiment_payload), encoding="utf-8")
    return model_path, experiment_path


def _build_command(
    *,
    model_path: Path,
    experiment_path: Path,
    record: dict,
    replay_log_file: Optional[str],
) -> list[str]:
    repro = record.get("repro", {})
    replay = repro.get("replay", {})
    seed_list = replay.get("seed_list") or repro.get("seeds")
    if not seed_list:
        raise ValueError("Record is missing repro seed information required for replay.")

    command = [
        sys.executable,
        "main.py",
        "--pool",
        record["pool"]["method"],
        "--pool-ratio",
        str(record["pool"]["ratio"]),
        "--dataset",
        record["dataset"],
        "--model-type",
        record.get("model", {}).get("variant", "sum"),
        "--model-config",
        str(model_path),
        "--experiment-config",
        str(experiment_path),
        "--seed-mode",
        "list",
        "--seed-list",
        ",".join(str(seed) for seed in seed_list),
    ]
    if record.get("comment") is not None:
        command.extend(["--comment", record["comment"]])
    if replay_log_file is not None:
        command.extend(["--log-file", replay_log_file])
    return command


def _stringify_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


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
    print(f"Expected split_digest: {record.get('repro', {}).get('split_digest', 'unknown')}")
    print(f"Expected protocol_digest: {record.get('repro', {}).get('protocol_digest', 'unknown')}")

    if run:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    app()
