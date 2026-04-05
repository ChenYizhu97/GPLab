import typer
from typing_extensions import Annotated

from experiment.config import build_request_from_normalized_job
from experiment.record import append_record_if_needed, summarize_record
from experiment.runner import run_experiment
from utils.jobs import load_job_file, normalize_complete_job_payload
from utils.presentation import build_error_payload, emit_json, validate_output_format

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    job_file: Annotated[str, typer.Option(..., help="Path to a complete automation job JSON file.")],
    output_format: Annotated[str, typer.Option(help="Output format: text or json.")] = "json",
):
    output_format = validate_output_format(output_format)
    try:
        loaded_job = load_job_file(job_file)
        normalized_job = normalize_complete_job_payload(loaded_job)
        conf, final_log_file, final_seed_mode, final_seed_base, final_allow_dup, final_seed_list = (
            build_request_from_normalized_job(normalized_job)
        )
        record = run_experiment(conf, emit_text=output_format == "text")
        append_record_if_needed(final_log_file, record)

        if output_format == "json":
            emit_json(
                {
                    "ok": True,
                    "kind": "train_result",
                    "record": record,
                    "summary": summarize_record(record),
                    "request": {
                        "job_file": job_file,
                        "mode": "strict_job",
                        "normalized_job": normalized_job,
                        "log_file": final_log_file,
                        "seed_mode": final_seed_mode,
                        "seed_base": final_seed_base,
                        "allow_duplicate_seeds": final_allow_dup,
                        "seed_list": final_seed_list,
                    },
                }
            )
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("train_error", exc, details={"job_file": job_file, "mode": "strict_job"}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
