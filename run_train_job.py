import typer
from typing_extensions import Annotated

from experiment.request_job import build_job_request
from experiment.train_result import execute_train_request
from utils.jobs import load_job_file, normalize_train_job
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
        normalized_job = normalize_train_job(loaded_job)
        request = build_job_request(normalized_job)
        payload = execute_train_request(
            request,
            emit_text=output_format == "text",
            request_details={
                "job_file": job_file,
                "mode": "strict_job",
                "normalized_job": normalized_job,
            },
        )

        if output_format == "json":
            emit_json(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        if output_format == "json":
            emit_json(build_error_payload("train_error", exc, details={"job_file": job_file, "mode": "strict_job"}))
            raise typer.Exit(code=1)
        raise


if __name__ == "__main__":
    app()
