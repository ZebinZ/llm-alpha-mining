from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


def canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        items: Sequence[Any] = tuple(value)
        if isinstance(value, (set, frozenset)):
            items = tuple(sorted(items, key=str))
        return [canonicalize(item) for item in items]
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise ValueError("naive timestamp cannot be canonically hashed")
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_frame(frame: pd.DataFrame) -> str:
    values = pd.DataFrame(frame)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(list(map(str, values.columns))))
    digest.update(canonical_json_bytes([str(dtype) for dtype in values.dtypes]))
    digest.update(pd.util.hash_pandas_object(values, index=True).values.tobytes())
    return digest.hexdigest()


def require_sha256(value: str, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return text


__all__ = [
    "canonical_json_bytes",
    "canonicalize",
    "hash_file",
    "hash_frame",
    "hash_json",
    "require_sha256",
]
