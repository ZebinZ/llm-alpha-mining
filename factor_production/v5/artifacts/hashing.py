from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class CanonicalJsonError(ValueError):
    pass


def _normalize(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError(f"non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"non-string object key at {path}")
            normalized[key] = _normalize(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item, path=f"{path}[]") for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize(value.to_dict(), path=path)
    raise CanonicalJsonError(f"unsupported canonical JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(value: Any) -> str:
    return hash_bytes(canonical_json_bytes(value))


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

