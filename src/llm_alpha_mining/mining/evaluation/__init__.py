"""Portable label and evaluation primitives, without private cache adapters."""

from .execution_labels import (
    ExecutionLabelPolicy,
    ExecutionLabelResult,
    build_execution_label_windows,
    compute_execution_aware_weekly_returns,
    select_execution_labels_between,
    validate_execution_label_windows,
)
from .gates import SubmissionGateDecision, evaluate_submission_gate
from .labels import ForwardReturnPolicy, ForwardReturnResult, compute_forward_returns
