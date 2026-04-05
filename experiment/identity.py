import hashlib
import json


def compute_record_id(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != "record_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def ensure_record_id(record: dict) -> dict:
    if "record_id" not in record:
        record["record_id"] = compute_record_id(record)
    return record


def compute_benchmark_key(record: dict) -> str:
    spec = record["spec"]
    payload = {
        "dataset": spec["dataset"],
        "model": spec["model"],
        "train": spec["train"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]
