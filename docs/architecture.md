# Architecture and research decisions

**English** | [简体中文](architecture.zh-CN.md) · [README](../README.md)

## Proposals are data, not executable model output

`mining/` owns candidate discovery and campaign state. `CandidateSpec` records the expression, direction, input fields, frequency, parents, and protocol hash. LLM roles return schema-validated objects. A deterministic panel applies their decisions, including a risk veto. The DSL walks an allowlisted syntax tree and dispatches registered operators; there is no Python `eval` entry point for model output.

`research/` binds an accepted candidate to a data snapshot, security contract, frequency, preprocessing rules, and operator registry. `factor_spec_from_candidate` supplies that boundary explicitly. A plausible formula alone is not an executable research specification.

## Point-in-time semantics

Observation time, availability time, and dataset version are distinct. `FactorDataView` separates instrument identity history from the universe eligible at the signal time. Historical observations can remain available for a rolling window even when a stock is outside today's cross-section. Non-stock instruments must never influence a stock cross-sectional rank. Forward returns are built as labels and cannot be factor inputs.

Missing observations and incomplete warm-up windows remain missing. A missing value is not automatically an economic zero. Availability contracts and time-based validation are necessary controls; they cannot establish that an external data source actually satisfies its own declarations.

## A lesson from the original panel correction

The source research required a snapshot-panel correction and recomputation of dependent factors. The methodological lesson is to distinguish an absent update from a valid zero, retain genuine abnormal-data checks, and avoid widening history requirements merely to improve coverage. A corrected input panel can change stock coverage, ranks, return diagnostics, and correlations. Previously selected factors therefore require reevaluation and correlation screening.

The public edition preserves relevant identity, masking, and computation contracts. Licensed Level 2 data, vendor-specific repair runners, and the resulting factor catalogs stay in the private research archive. This repository cannot reproduce that private panel reconstruction by itself.

## Evaluation and selection

Labels, temporal splits, factor metrics, portfolio construction, costs, and execution are separate components. Evaluation should use frozen definitions of the forecast horizon, trade timing, universe, and cost assumptions. The significance module can evaluate a frozen candidate family and apply multiple-testing corrections. These checks support a research protocol; passing tests or examining many formulas does not establish predictive value.

The default demo computes descriptive correlations and coverage only. Formal temporal evaluation, backtesting, and family-level significance have dedicated tests. External out-of-sample results must remain outside iterative search feedback.

## Recoverable execution

`mining/orchestration/` maintains candidate states, lineage, evaluation attempts, stopping conditions, and SQLite persistence. The generation executor records stage checkpoints and checks protocol and artifact identities during recovery. Shared content caches identify computations by their inputs rather than by a mutable filename.

Structured LLM calls have budgets, request identities, and a call ledger. An ambiguous network failure is retained as uncertain rather than silently repeated. Exact replay checks the request identity. Live transport is optional; the default demo injects a scripted transport and makes no network calls.

## Package and artifact identity

The public API uses the `llm_alpha_mining` namespace. `mining` contains discovery and execution; `research` contains scientific computation; `demo` composes a small reproducible example. Tests use component names rather than project-phase numbers.

Some serialized schemas retain suffixes such as `/v5`, and versioned operator semantics remain available for replay. These are data-format identities, not duplicate source trees. Changing their meaning without changing their identity would make hashes misleading.

Artifacts connect expressions, data, metrics, and signal values. The demo verifies files against a SHA-256 manifest. Hash equality establishes byte integrity; it does not establish sound methodology or investment performance.
