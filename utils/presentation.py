import json
from pathlib import Path
from typing import Any, Optional

import typer


OUTPUT_FORMATS = ("text", "json")


def validate_output_format(value: str) -> str:
    if value not in OUTPUT_FORMATS:
        raise typer.BadParameter(
            f"output_format must be one of: {', '.join(OUTPUT_FORMATS)}.",
            param_hint="--output-format",
        )
    return value


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def build_error_payload(kind: str, exc: Exception, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    error_type = "runtime_error"
    if isinstance(exc, typer.BadParameter) or isinstance(exc, ValueError):
        error_type = "config_error"
    elif isinstance(exc, FileNotFoundError):
        error_type = "file_not_found"

    payload: dict[str, Any] = {
        "ok": False,
        "kind": kind,
        "error": {
            "type": error_type,
            "message": str(exc),
        },
    }
    if details:
        payload["error"]["details"] = details
    return payload


def ensure_parent_dir(path: Optional[str]) -> None:
    if path is None:
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
