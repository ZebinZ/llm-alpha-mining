from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

import numpy as np
import pandas as pd

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_frame,
    hash_json,
    require_sha256,
)
from alpha_research.evaluation import FactorEvaluationReport
from alpha_research.experiments import DataPartition
from alpha_research.market_logic.evidence import EvidenceVisibility


VALIDATION_AUDIT_ARTIFACT_SCHEMA = "validation-audit-artifact/v1"
VALIDATION_AUDIT_BINDING_SCHEMA = "validation-audit-binding/v1"
VALIDATION_AUDIT_RECEIPT_SCHEMA = "validation-audit-receipt/v1"
_FRAME_SCHEMA = "canonical-pandas-frame/v1"
_OBJECT_PARTS = ("audit_only", "validation_reports")
_MEDIA_TYPE = "application/vnd.alpha-research.validation-audit+json"


class ValidationAuditArtifactFailure(str, Enum):
    """Metric-free failure codes safe for workflow logs."""

    INVALID_BINDING = "invalid_binding"
    UNSUPPORTED_REPORT = "unsupported_report"
    REPORT_INTEGRITY_FAILED = "report_integrity_failed"
    UNSAFE_STORE_PATH = "unsafe_store_path"
    CONTENT_CONFLICT = "content_conflict"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_INTEGRITY_FAILED = "artifact_integrity_failed"
    PAYLOAD_INVALID = "payload_invalid"
    ARTIFACT_TOO_LARGE = "artifact_too_large"


class ValidationAuditArtifactError(RuntimeError):
    """Fail-closed audit-store error that never includes metrics or paths."""

    def __init__(self, code: ValidationAuditArtifactFailure) -> None:
        self.code = ValidationAuditArtifactFailure(code)
        super().__init__(self.code.value)


@dataclass(frozen=True, slots=True)
class ValidationAuditArtifactReceipt:
    """Public descriptor for an exact report held outside adaptive memory.

    The descriptor deliberately contains only immutable hashes, a safe relative
    location, byte size, media type, and the two fixed governance labels.  It
    contains no metric, date, security, verdict, or free-text field.
    """

    evaluation_report_hash: str
    artifact_hash: str
    binding_hash: str
    experiment_hash: str
    logic_hash: str
    relative_location: str
    size_bytes: int
    source_partition: DataPartition | str = DataPartition.VALIDATION
    visibility: EvidenceVisibility | str = EvidenceVisibility.AUDIT_ONLY
    media_type: str = _MEDIA_TYPE
    schema_version: str = VALIDATION_AUDIT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_AUDIT_RECEIPT_SCHEMA:
            raise ValueError("unsupported validation audit receipt schema")
        for name in (
            "evaluation_report_hash",
            "artifact_hash",
            "binding_hash",
            "experiment_hash",
            "logic_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"audit receipt {name}")
        partition = DataPartition(self.source_partition)
        visibility = EvidenceVisibility(self.visibility)
        if partition is not DataPartition.VALIDATION:
            raise ValueError("audit receipt source partition must be validation")
        if visibility is not EvidenceVisibility.AUDIT_ONLY:
            raise ValueError("audit receipt visibility must be audit-only")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes <= 0
        ):
            raise ValueError("audit receipt size must be positive")
        if self.media_type != _MEDIA_TYPE:
            raise ValueError("audit receipt media type differs")
        expected_location = _relative_location(self.evaluation_report_hash)
        if self.relative_location != expected_location:
            raise ValueError("audit receipt location differs from content address")
        path = PurePosixPath(self.relative_location)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("audit receipt location is unsafe")
        if self.binding_hash != _binding_hash(
            experiment_hash=self.experiment_hash,
            logic_hash=self.logic_hash,
            evaluation_report_hash=self.evaluation_report_hash,
        ):
            raise ValueError("audit receipt binding hash differs")
        object.__setattr__(self, "source_partition", partition)
        object.__setattr__(self, "visibility", visibility)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_report_hash": self.evaluation_report_hash,
            "artifact_hash": self.artifact_hash,
            "binding_hash": self.binding_hash,
            "experiment_hash": self.experiment_hash,
            "logic_hash": self.logic_hash,
            "relative_location": self.relative_location,
            "size_bytes": self.size_bytes,
            "source_partition": DataPartition(self.source_partition).value,
            "visibility": EvidenceVisibility(self.visibility).value,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "ValidationAuditArtifactReceipt":
        expected = {
            "schema_version",
            "evaluation_report_hash",
            "artifact_hash",
            "binding_hash",
            "experiment_hash",
            "logic_hash",
            "relative_location",
            "size_bytes",
            "source_partition",
            "visibility",
            "media_type",
        }
        if set(payload) != expected:
            raise ValueError("validation audit receipt fields differ")
        string_fields = expected.difference({"size_bytes"})
        if not all(isinstance(payload[name], str) for name in string_fields):
            raise TypeError("validation audit receipt scalar fields must be strings")
        size = payload["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("validation audit receipt size must be an integer")
        return cls(
            schema_version=str(payload["schema_version"]),
            evaluation_report_hash=str(payload["evaluation_report_hash"]),
            artifact_hash=str(payload["artifact_hash"]),
            binding_hash=str(payload["binding_hash"]),
            experiment_hash=str(payload["experiment_hash"]),
            logic_hash=str(payload["logic_hash"]),
            relative_location=str(payload["relative_location"]),
            size_bytes=size,
            source_partition=str(payload["source_partition"]),
            visibility=str(payload["visibility"]),
            media_type=str(payload["media_type"]),
        )


class ValidationEvaluationAuditStore:
    """Write-once, content-addressed store for exact validation reports.

    This store is intentionally independent of ``MarketLogicRegistry``.  Its
    only read API is named ``load_for_audit`` and returns the sealed exact
    ``FactorEvaluationReport``; neither this store nor its receipt implements a
    research-memory retrieval interface.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        maximum_artifact_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if (
            not isinstance(maximum_artifact_bytes, int)
            or isinstance(maximum_artifact_bytes, bool)
            or maximum_artifact_bytes <= 0
        ):
            raise ValueError("maximum_artifact_bytes must be positive")
        requested = Path(os.path.abspath(os.fspath(root)))
        try:
            _reject_existing_symlink_components(requested)
            requested.mkdir(parents=True, exist_ok=True, mode=0o750)
            _reject_existing_symlink_components(requested)
            identity = os.stat(requested, follow_symlinks=False)
        except (OSError, ValueError) as exc:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
            ) from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
            )
        self.root = requested
        self.maximum_artifact_bytes = maximum_artifact_bytes

    def publish(
        self,
        report: FactorEvaluationReport,
        *,
        experiment_hash: str,
        logic_hash: str,
        source_partition: DataPartition | str = DataPartition.VALIDATION,
        visibility: EvidenceVisibility | str = EvidenceVisibility.AUDIT_ONLY,
    ) -> ValidationAuditArtifactReceipt:
        experiment, logic = _validate_binding(
            experiment_hash=experiment_hash,
            logic_hash=logic_hash,
            source_partition=source_partition,
            visibility=visibility,
        )
        if type(report) is not FactorEvaluationReport:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.UNSUPPORTED_REPORT
            )
        try:
            report_payload = _encode_report(report)
            # Reconstruct before writing.  This catches a torn snapshot if a
            # caller mutates one of the DataFrames concurrently with sealing.
            reconstructed = _decode_report(report_payload)
            if reconstructed.content_hash != report.content_hash:
                raise ValueError("report changed while it was sealed")
            binding_hash = _binding_hash(
                experiment_hash=experiment,
                logic_hash=logic,
                evaluation_report_hash=reconstructed.content_hash,
            )
            envelope = {
                "schema_version": VALIDATION_AUDIT_ARTIFACT_SCHEMA,
                "binding": {
                    "schema_version": VALIDATION_AUDIT_BINDING_SCHEMA,
                    "experiment_hash": experiment,
                    "logic_hash": logic,
                    "evaluation_report_hash": reconstructed.content_hash,
                    "source_partition": DataPartition.VALIDATION.value,
                    "visibility": EvidenceVisibility.AUDIT_ONLY.value,
                    "binding_hash": binding_hash,
                },
                "report": report_payload,
            }
            payload = canonical_json_bytes(envelope)
        except ValidationAuditArtifactError:
            raise
        except Exception as exc:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.REPORT_INTEGRITY_FAILED
            ) from exc
        if len(payload) > self.maximum_artifact_bytes:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.ARTIFACT_TOO_LARGE
            )
        artifact_hash = _sha256_bytes(payload)
        relative = _relative_location(reconstructed.content_hash)
        self._write_once(
            report_hash=reconstructed.content_hash,
            payload=payload,
        )
        return ValidationAuditArtifactReceipt(
            evaluation_report_hash=reconstructed.content_hash,
            artifact_hash=artifact_hash,
            binding_hash=binding_hash,
            experiment_hash=experiment,
            logic_hash=logic,
            relative_location=relative,
            size_bytes=len(payload),
        )

    def load_for_audit(
        self, receipt: ValidationAuditArtifactReceipt
    ) -> FactorEvaluationReport:
        if type(receipt) is not ValidationAuditArtifactReceipt:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.INVALID_BINDING
            )
        if receipt.size_bytes > self.maximum_artifact_bytes:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.ARTIFACT_TOO_LARGE
            )
        payload = self._read_existing(
            report_hash=receipt.evaluation_report_hash,
            expected_size=receipt.size_bytes,
        )
        if _sha256_bytes(payload) != receipt.artifact_hash:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.ARTIFACT_INTEGRITY_FAILED
            )
        try:
            decoded = json.loads(payload.decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise TypeError("audit envelope must be an object")
            if canonical_json_bytes(decoded) != payload:
                raise ValueError("audit envelope is not canonical JSON")
            if set(decoded) != {"schema_version", "binding", "report"}:
                raise ValueError("audit envelope fields differ")
            if decoded["schema_version"] != VALIDATION_AUDIT_ARTIFACT_SCHEMA:
                raise ValueError("audit envelope schema differs")
            binding = decoded["binding"]
            if not isinstance(binding, Mapping):
                raise TypeError("audit binding must be an object")
            _verify_envelope_binding(binding, receipt)
            report_payload = decoded["report"]
            if not isinstance(report_payload, Mapping):
                raise TypeError("audit report must be an object")
            report = _decode_report(report_payload)
            if report.content_hash != receipt.evaluation_report_hash:
                raise ValueError("audit report content hash differs")
            return report
        except ValidationAuditArtifactError:
            raise
        except Exception as exc:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.PAYLOAD_INVALID
            ) from exc

    def _write_once(self, *, report_hash: str, payload: bytes) -> None:
        root_fd = self._open_root()
        descriptors: list[int] = [root_fd]
        created_fd: int | None = None
        leaf = f"{report_hash}.json"
        try:
            parent_fd = root_fd
            for name in (*_OBJECT_PARTS, report_hash[:2]):
                child_fd = _open_directory(parent_fd, name, create=True)
                descriptors.append(child_fd)
                parent_fd = child_fd
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                created_fd = os.open(leaf, flags, 0o440, dir_fd=parent_fd)
            except FileExistsError:
                existing = _read_regular_file(
                    parent_fd,
                    leaf,
                    maximum_bytes=self.maximum_artifact_bytes,
                )
                if existing != payload:
                    raise ValidationAuditArtifactError(
                        ValidationAuditArtifactFailure.CONTENT_CONFLICT
                    ) from None
                return
            _write_all(created_fd, payload)
            os.fsync(created_fd)
            os.fchmod(created_fd, 0o440)
            os.fsync(parent_fd)
        except ValidationAuditArtifactError:
            if created_fd is not None:
                _unlink_if_same_file(descriptors[-1], leaf, created_fd)
            raise
        except OSError as exc:
            if created_fd is not None:
                _unlink_if_same_file(descriptors[-1], leaf, created_fd)
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
            ) from exc
        finally:
            if created_fd is not None:
                os.close(created_fd)
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _read_existing(self, *, report_hash: str, expected_size: int) -> bytes:
        root_fd = self._open_root()
        descriptors: list[int] = [root_fd]
        try:
            parent_fd = root_fd
            for name in (*_OBJECT_PARTS, report_hash[:2]):
                child_fd = _open_directory(parent_fd, name, create=False)
                descriptors.append(child_fd)
                parent_fd = child_fd
            payload = _read_regular_file(
                parent_fd,
                f"{report_hash}.json",
                maximum_bytes=self.maximum_artifact_bytes,
            )
            if len(payload) != expected_size:
                raise ValidationAuditArtifactError(
                    ValidationAuditArtifactFailure.ARTIFACT_INTEGRITY_FAILED
                )
            return payload
        except FileNotFoundError as exc:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.ARTIFACT_NOT_FOUND
            ) from exc
        except ValidationAuditArtifactError:
            raise
        except OSError as exc:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
            ) from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _open_root(self) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            _reject_existing_symlink_components(self.root)
            descriptor = os.open(self.root, flags)
            identity = os.fstat(descriptor)
            if not stat.S_ISDIR(identity.st_mode):
                os.close(descriptor)
                raise OSError("audit root is not a directory")
            return descriptor
        except (OSError, ValueError) as exc:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
            ) from exc


def _validate_binding(
    *,
    experiment_hash: str,
    logic_hash: str,
    source_partition: DataPartition | str,
    visibility: EvidenceVisibility | str,
) -> tuple[str, str]:
    try:
        experiment = require_sha256(str(experiment_hash), name="audit experiment hash")
        logic = require_sha256(str(logic_hash), name="audit logic hash")
        partition = DataPartition(source_partition)
        exposed = EvidenceVisibility(visibility)
    except (TypeError, ValueError) as exc:
        raise ValidationAuditArtifactError(
            ValidationAuditArtifactFailure.INVALID_BINDING
        ) from exc
    if (
        partition is not DataPartition.VALIDATION
        or exposed is not EvidenceVisibility.AUDIT_ONLY
    ):
        raise ValidationAuditArtifactError(
            ValidationAuditArtifactFailure.INVALID_BINDING
        )
    return experiment, logic


def _binding_hash(
    *, experiment_hash: str, logic_hash: str, evaluation_report_hash: str
) -> str:
    return hash_json(
        {
            "schema_version": VALIDATION_AUDIT_BINDING_SCHEMA,
            "experiment_hash": experiment_hash,
            "logic_hash": logic_hash,
            "evaluation_report_hash": evaluation_report_hash,
            "source_partition": DataPartition.VALIDATION.value,
            "visibility": EvidenceVisibility.AUDIT_ONLY.value,
        }
    )


def _verify_envelope_binding(
    payload: Mapping[str, object], receipt: ValidationAuditArtifactReceipt
) -> None:
    expected = {
        "schema_version",
        "experiment_hash",
        "logic_hash",
        "evaluation_report_hash",
        "source_partition",
        "visibility",
        "binding_hash",
    }
    if set(payload) != expected or not all(
        isinstance(payload[name], str) for name in expected
    ):
        raise ValueError("audit binding fields differ")
    observed = {
        "schema_version": VALIDATION_AUDIT_BINDING_SCHEMA,
        "experiment_hash": receipt.experiment_hash,
        "logic_hash": receipt.logic_hash,
        "evaluation_report_hash": receipt.evaluation_report_hash,
        "source_partition": DataPartition.VALIDATION.value,
        "visibility": EvidenceVisibility.AUDIT_ONLY.value,
        "binding_hash": receipt.binding_hash,
    }
    if dict(payload) != observed:
        raise ValueError("audit envelope binding differs from receipt")


def _encode_report(report: FactorEvaluationReport) -> dict[str, object]:
    # Reading and reconstructing all four content hashes below is the sealing
    # operation.  FactorEvaluationReport copies frames at construction time but
    # pandas objects remain mutable, so callers cannot rely on frozen dataclass
    # syntax alone.
    report_hash = report.content_hash
    payload: dict[str, object] = {
        "schema_version": report.schema_version,
        "evaluation_spec_hash": report.evaluation_spec_hash,
        "factor_spec_hash": report.factor_spec_hash,
        "factor_signal_hash": report.factor_signal_hash,
        "label_spec_hash": report.label_spec_hash,
        "label_values_hash": report.label_values_hash,
        "validation_receipt_hash": report.validation_receipt_hash,
        "reference_factor_hashes": dict(report.reference_factor_hashes),
        "decay_label_result_hashes": dict(report.decay_label_result_hashes),
        "fold_id": report.fold_id,
        "per_date_hash": report.per_date_hash,
        "quantile_returns_hash": report.quantile_returns_hash,
        "summary_hash": report.summary_hash,
        "metric_metadata_hash": report.metric_metadata_hash,
        "per_date": _encode_frame(report.per_date),
        "quantile_returns": _encode_frame(report.quantile_returns),
        "summary": [
            [name, _encode_scalar(value)]
            for name, value in sorted(report.summary.items())
        ],
        "metric_metadata": _encode_frame(report.metric_metadata),
        "report_content_hash": report_hash,
    }
    return payload


def _decode_report(payload: Mapping[str, object]) -> FactorEvaluationReport:
    expected = {
        "schema_version",
        "evaluation_spec_hash",
        "factor_spec_hash",
        "factor_signal_hash",
        "label_spec_hash",
        "label_values_hash",
        "validation_receipt_hash",
        "reference_factor_hashes",
        "decay_label_result_hashes",
        "fold_id",
        "per_date_hash",
        "quantile_returns_hash",
        "summary_hash",
        "metric_metadata_hash",
        "per_date",
        "quantile_returns",
        "summary",
        "metric_metadata",
        "report_content_hash",
    }
    if set(payload) != expected:
        raise ValueError("evaluation report payload fields differ")
    scalar_fields = {
        "schema_version",
        "evaluation_spec_hash",
        "factor_spec_hash",
        "factor_signal_hash",
        "label_spec_hash",
        "label_values_hash",
        "validation_receipt_hash",
        "fold_id",
        "per_date_hash",
        "quantile_returns_hash",
        "summary_hash",
        "metric_metadata_hash",
        "report_content_hash",
    }
    if not all(isinstance(payload[name], str) for name in scalar_fields):
        raise TypeError("evaluation report scalar fields must be strings")
    for name in scalar_fields.difference({"schema_version", "fold_id"}):
        require_sha256(str(payload[name]), name=f"evaluation report {name}")
    references = _hash_mapping(payload["reference_factor_hashes"])
    decay = _hash_mapping(payload["decay_label_result_hashes"])
    raw_summary = payload["summary"]
    if not isinstance(raw_summary, list):
        raise TypeError("evaluation report summary must be a list")
    summary: dict[str, float | int | None] = {}
    previous: str | None = None
    for item in raw_summary:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], Mapping)
        ):
            raise TypeError("evaluation report summary item is invalid")
        name = item[0]
        if previous is not None and name <= previous:
            raise ValueError("evaluation report summary keys are not canonical")
        value = _decode_scalar(item[1])
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise TypeError("evaluation report metric has an unsupported type")
        summary[name] = value
        previous = name
    per_date_payload = payload["per_date"]
    quantile_payload = payload["quantile_returns"]
    metadata_payload = payload["metric_metadata"]
    if not all(
        isinstance(value, Mapping)
        for value in (per_date_payload, quantile_payload, metadata_payload)
    ):
        raise TypeError("evaluation report frame payload is invalid")
    report = FactorEvaluationReport(
        schema_version=str(payload["schema_version"]),
        evaluation_spec_hash=str(payload["evaluation_spec_hash"]),
        factor_spec_hash=str(payload["factor_spec_hash"]),
        factor_signal_hash=str(payload["factor_signal_hash"]),
        label_spec_hash=str(payload["label_spec_hash"]),
        label_values_hash=str(payload["label_values_hash"]),
        validation_receipt_hash=str(payload["validation_receipt_hash"]),
        reference_factor_hashes=references,
        decay_label_result_hashes=decay,
        fold_id=str(payload["fold_id"]),
        per_date_hash=str(payload["per_date_hash"]),
        quantile_returns_hash=str(payload["quantile_returns_hash"]),
        summary_hash=str(payload["summary_hash"]),
        metric_metadata_hash=str(payload["metric_metadata_hash"]),
        per_date=_decode_frame(cast(Mapping[str, object], per_date_payload)),
        quantile_returns=_decode_frame(cast(Mapping[str, object], quantile_payload)),
        summary=summary,
        metric_metadata=_decode_frame(cast(Mapping[str, object], metadata_payload)),
    )
    if report.content_hash != payload["report_content_hash"]:
        raise ValueError("evaluation report content hash differs")
    return report


def _hash_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise TypeError("evaluation report hash mapping is invalid")
    result = dict(value)
    if list(result) != sorted(result) or any(not name for name in result):
        raise ValueError("evaluation report hash mapping is not canonical")
    for name, digest in result.items():
        require_sha256(digest, name=f"evaluation report mapping:{name}")
    return result


def _encode_frame(frame: pd.DataFrame) -> dict[str, object]:
    value = pd.DataFrame(frame)
    if isinstance(value.index, pd.MultiIndex):
        raise TypeError("multi-index evaluation artifacts are unsupported")
    if not all(isinstance(column, str) for column in value.columns):
        raise TypeError("evaluation artifact columns must be strings")
    columns = [str(column) for column in value.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("evaluation artifact columns must be unique")
    if value.index.name is not None and not isinstance(value.index.name, str):
        raise TypeError("evaluation artifact index name must be a string or null")
    return {
        "schema_version": _FRAME_SCHEMA,
        "frame_hash": hash_frame(value),
        "columns": columns,
        "dtypes": [str(dtype) for dtype in value.dtypes],
        "index": {
            "name": value.index.name,
            "dtype": str(value.index.dtype),
            "values": [_encode_scalar(item) for item in value.index],
        },
        "data": [
            [_encode_scalar(value.iat[row, column]) for column in range(len(columns))]
            for row in range(len(value))
        ],
    }


def _decode_frame(payload: Mapping[str, object]) -> pd.DataFrame:
    expected = {
        "schema_version",
        "frame_hash",
        "columns",
        "dtypes",
        "index",
        "data",
    }
    if set(payload) != expected or payload["schema_version"] != _FRAME_SCHEMA:
        raise ValueError("canonical frame fields differ")
    frame_hash = payload["frame_hash"]
    if not isinstance(frame_hash, str):
        raise TypeError("canonical frame hash must be a string")
    require_sha256(frame_hash, name="canonical frame hash")
    columns = payload["columns"]
    dtypes = payload["dtypes"]
    data = payload["data"]
    raw_index = payload["index"]
    if (
        not isinstance(columns, list)
        or not all(isinstance(item, str) for item in columns)
        or len(columns) != len(set(columns))
        or not isinstance(dtypes, list)
        or not all(isinstance(item, str) for item in dtypes)
        or len(columns) != len(dtypes)
        or not isinstance(data, list)
        or not isinstance(raw_index, Mapping)
    ):
        raise TypeError("canonical frame structure is invalid")
    if set(raw_index) != {"name", "dtype", "values"}:
        raise ValueError("canonical frame index fields differ")
    index_name = raw_index["name"]
    index_dtype = raw_index["dtype"]
    index_values = raw_index["values"]
    if (
        (index_name is not None and not isinstance(index_name, str))
        or not isinstance(index_dtype, str)
        or not isinstance(index_values, list)
        or not all(isinstance(item, Mapping) for item in index_values)
        or len(index_values) != len(data)
    ):
        raise TypeError("canonical frame index is invalid")
    decoded_rows: list[list[object]] = []
    for row in data:
        if (
            not isinstance(row, list)
            or len(row) != len(columns)
            or not all(isinstance(item, Mapping) for item in row)
        ):
            raise TypeError("canonical frame row is invalid")
        decoded_rows.append([_decode_scalar(item) for item in row])
    decoded_index = [_decode_scalar(item) for item in index_values]
    index = _build_index(decoded_index, dtype=index_dtype, name=index_name)
    arrays: dict[str, object] = {}
    for offset, (column, dtype) in enumerate(zip(columns, dtypes, strict=True)):
        arrays[column] = pd.array([row[offset] for row in decoded_rows], dtype=dtype)
    frame = pd.DataFrame(arrays, columns=columns)
    frame.index = index
    if [str(dtype) for dtype in frame.dtypes] != dtypes:
        raise ValueError("canonical frame dtypes differ")
    if hash_frame(frame) != frame_hash:
        raise ValueError("canonical frame content hash differs")
    return frame


def _build_index(values: list[object], *, dtype: str, name: str | None) -> pd.Index:
    if dtype.startswith("datetime64["):
        index: pd.Index = pd.DatetimeIndex(values, name=name)
    elif dtype.startswith("timedelta64["):
        index = pd.TimedeltaIndex(values, name=name)
    else:
        index = pd.Index(pd.array(values, dtype=dtype), name=name)
    if str(index.dtype) != dtype:
        raise ValueError("canonical frame index dtype differs")
    return index


def _encode_scalar(value: object) -> dict[str, object]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "null"}
    if isinstance(value, pd.Timestamp):
        return {
            "type": "timestamp_ns",
            "value": int(value.value),
            "timezone": None if value.tz is None else str(value.tz),
        }
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return {"type": "timedelta_ns", "value": int(pd.Timedelta(value).value)}
    if isinstance(value, (np.datetime64,)):
        timestamp = pd.Timestamp(value)
        return {
            "type": "timestamp_ns",
            "value": int(timestamp.value),
            "timezone": None,
        }
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, Integral):
        return {"type": "int", "value": int(value)}
    if isinstance(value, Real):
        number = float(value)
        # float.hex is deterministic, exact, and represents NaN/inf without
        # asking JSON to accept its non-standard numeric extensions.
        return {"type": "float64", "value": number.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError("evaluation artifact contains an unsupported scalar")


def _decode_scalar(payload: Mapping[str, object]) -> object:
    kind = payload.get("type")
    if kind == "null" and set(payload) == {"type"}:
        return None
    if kind == "bool" and set(payload) == {"type", "value"}:
        value = payload["value"]
        if type(value) is not bool:
            raise TypeError("boolean scalar is invalid")
        return value
    if kind == "int" and set(payload) == {"type", "value"}:
        value = payload["value"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("integer scalar is invalid")
        return value
    if kind == "float64" and set(payload) == {"type", "value"}:
        value = payload["value"]
        if not isinstance(value, str):
            raise TypeError("float scalar is invalid")
        return float.fromhex(value)
    if kind == "str" and set(payload) == {"type", "value"}:
        value = payload["value"]
        if not isinstance(value, str):
            raise TypeError("string scalar is invalid")
        return value
    if kind == "timedelta_ns" and set(payload) == {"type", "value"}:
        value = payload["value"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("timedelta scalar is invalid")
        return pd.Timedelta(value, unit="ns")
    if kind == "timestamp_ns" and set(payload) == {"type", "value", "timezone"}:
        value = payload["value"]
        timezone = payload["timezone"]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or (timezone is not None and not isinstance(timezone, str))
        ):
            raise TypeError("timestamp scalar is invalid")
        timestamp = pd.Timestamp(value, unit="ns", tz="UTC")
        return (
            timestamp.tz_localize(None)
            if timezone is None
            else timestamp.tz_convert(timezone)
        )
    raise ValueError("unsupported canonical scalar")


def _relative_location(report_hash: str) -> str:
    require_sha256(report_hash, name="audit report hash")
    return "/".join((*_OBJECT_PARTS, report_hash[:2], f"{report_hash}.json"))


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _reject_existing_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            identity = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(identity.st_mode):
            raise ValueError("symbolic-link path components are forbidden")


def _open_directory(parent_fd: int, name: str, *, create: bool) -> int:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValidationAuditArtifactError(
            ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
        )
    if create:
        try:
            os.mkdir(name, 0o750, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
    identity = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(identity.st_mode):
        raise ValidationAuditArtifactError(
            ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(name, flags, dir_fd=parent_fd)


def _read_regular_file(parent_fd: int, name: str, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_nlink != 1
            or identity.st_size <= 0
        ):
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.UNSAFE_STORE_PATH
            )
        if identity.st_size > maximum_bytes:
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.ARTIFACT_TOO_LARGE
            )
        chunks: list[bytes] = []
        remaining = identity.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValidationAuditArtifactError(
                    ValidationAuditArtifactFailure.ARTIFACT_INTEGRITY_FAILED
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValidationAuditArtifactError(
                ValidationAuditArtifactFailure.ARTIFACT_INTEGRITY_FAILED
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short audit artifact write")
        view = view[written:]


def _unlink_if_same_file(parent_fd: int, name: str, descriptor: int) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino):
            os.unlink(name, dir_fd=parent_fd)
    except OSError:
        pass


__all__ = [
    "VALIDATION_AUDIT_ARTIFACT_SCHEMA",
    "VALIDATION_AUDIT_BINDING_SCHEMA",
    "VALIDATION_AUDIT_RECEIPT_SCHEMA",
    "ValidationAuditArtifactError",
    "ValidationAuditArtifactFailure",
    "ValidationAuditArtifactReceipt",
    "ValidationEvaluationAuditStore",
]
