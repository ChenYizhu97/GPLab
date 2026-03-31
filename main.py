import toml
import typer
from experiment.config import build_experiment_config
from experiment.record import append_record_if_needed
from experiment.runner import run_experiment
from utils.cli import resolve_seed_options
from typing_extensions import Annotated, Optional

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
        pool: Annotated[str, typer.Option(help="Pooling method name or <module:factory> for custom pooling.")] = "nopool",
        pool_ratio: Annotated[float, typer.Option(help="Pooling ratio for built-in or custom pooling methods.")] = 0.5,
        dataset: Annotated[str, typer.Option()] = "PROTEINS",
        model_type: Annotated[str, typer.Option(help="Model type: sum or plain.")] = "sum",
        log_file: Annotated[Optional[str], typer.Option(help="JSONL file path to append experiment records.")] = None,
        model_config: Annotated[str, typer.Option()] = "config/model.toml",
        experiment_config: Annotated[str, typer.Option()] = "config/experiment.toml",
        comment: Annotated[Optional[str], typer.Option()] = None,
        seed_mode: Annotated[Optional[str], typer.Option(help="Seed source mode: auto, file, or list.")] = None,
        seed_base: Annotated[Optional[int], typer.Option(help="Base integer for deterministic seed generation in auto mode.")] = None,
        seed_list: Annotated[Optional[str], typer.Option(help="Comma-separated seed list for exact replay, for example: 11,22,33")] = None,
        allow_duplicate_seeds: Annotated[Optional[bool], typer.Option(help="Allow duplicate seeds in file or list mode.")] = None,
):
    model_conf = toml.load(model_config)
    experiment_conf = toml.load(experiment_config)

    exp = experiment_conf.get("experiment", {})
    final_seed_mode, final_seed_base, final_allow_dup, final_seed_list = resolve_seed_options(
        seed_mode=seed_mode,
        seed_base=seed_base,
        seed_list=seed_list,
        allow_duplicate_seeds=allow_duplicate_seeds,
        expr_conf=exp,
    )

    conf = build_experiment_config(
        model_conf=model_conf,
        experiment_conf=experiment_conf,
        pool=pool,
        pool_ratio=pool_ratio,
        dataset_name=dataset,
        model_type=model_type,
        comment=comment,
        seed_mode=final_seed_mode,
        seed_base=final_seed_base,
        seed_list=final_seed_list,
        allow_duplicate_seeds=final_allow_dup,
    )
    record = run_experiment(conf)
    append_record_if_needed(log_file, record)


if __name__ == "__main__":
    app()
