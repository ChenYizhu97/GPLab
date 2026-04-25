from typing import Any, Optional

from gplab.experiment.execute import execute_request
from gplab.experiment.record import summarize_record
from gplab.experiment.request_common import TrainRequestContext
from gplab.utils.jsonl import append_jsonl


def execute_train_request(
    request: TrainRequestContext,
    *,
    emit_text: bool,
    request_details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record = execute_request(request.conf, emit_text=emit_text)
    if request.log_file is not None:
        append_jsonl(request.log_file, record)

    request_payload: dict[str, Any] = {}
    if request_details:
        request_payload.update(request_details)
    request_payload.update(
        {
            "log_file": request.log_file,
            "seed_mode": request.seed_mode,
            "seed_base": request.seed_base,
            "allow_duplicate_seeds": request.allow_duplicate_seeds,
            "seed_list": request.seed_list,
        }
    )

    return {
        "ok": True,
        "kind": "train_result",
        "record": record,
        "summary": summarize_record(record),
        "request": request_payload,
    }
