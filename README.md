# LLM Alpha Mining

**English** | [简体中文](README.zh-CN.md)

[![Tests](https://github.com/ZebinZ/llm-alpha-mining/actions/workflows/tests.yml/badge.svg)](https://github.com/ZebinZ/llm-alpha-mining/actions/workflows/tests.yml)

**A quantitative research framework that turns LLM proposals into constrained factor definitions, point-in-time signals, and reproducible research artifacts.**

The central question is how to make automated factor discovery inspectable: which hypothesis was proposed, what data could it see, why was it accepted, and can its signal be reproduced? Language models propose and review candidates; deterministic code controls expression validity, risk vetoes, data alignment, evaluation, and provenance.

This repository contains the reusable framework developed during a quantitative research project. It includes an offline example, 251 automated tests, and English and Chinese documentation. Research data and individual factor submissions are kept outside the repository. This project uses language models for research assistance; it does not train a foundation model.

## Start here

Use Python 3.12 or 3.13. Installation downloads dependencies. The example runs offline with no API key.

```bash
git clone https://github.com/ZebinZ/llm-alpha-mining.git
cd llm-alpha-mining
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
alpha-demo --output outputs/demo
alpha-demo --output outputs/demo --verify
python -m pytest -q
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Each demo run needs a new output directory; `--verify` checks an existing run.

The example generates 100 business days for 40 fictional stocks plus a fictional index. Scripted role responses propose three formulas; one receives a risk veto and two reach the real factor engine. It exercises missing observations, rolling-window warm-up, a changing trading universe, and exclusion of non-stock instruments.

Expected outcome: two accepted factors, zero network calls, and `PASS` from artifact verification. The output includes signal Parquet files, descriptive diagnostics, the call ledger, `report.json`, and a SHA-256 `manifest.json`. These are software checks on synthetic data, not evidence of investment performance.

## What the framework does

```mermaid
flowchart LR
    A[Research hypotheses] --> B[Structured LLM proposals]
    B --> C[DSL validation and role review]
    C --> D[Point-in-time factor computation]
    D --> E[Temporal validation and cost evaluation]
    E --> F[Selection and reproducible artifacts]
```

| Capability | Implementation | Evidence to inspect |
| --- | --- | --- |
| Structured proposal, critic, risk, and arbiter roles | [Mining protocols and LLM calls](src/llm_alpha_mining/mining/llm) | [Structured LLM tests](tests/test_structured_llm.py) |
| Formula allowlists and restrictions on future information | [DSL interpreter](src/llm_alpha_mining/mining/dsl/interpreter.py) | [DSL tests](tests/test_safe_dsl.py) |
| Separation of instrument identity, available history, and signal-time universe | [Factor engine](src/llm_alpha_mining/research/factors/engine.py) | [Factor contract tests](tests/test_factor_contracts.py) |
| Labels, temporal splits, and factor evaluation | [Research evaluation](src/llm_alpha_mining/research/evaluation) | [Label and validation tests](tests/test_labels_and_validation.py), [evaluation tests](tests/test_evaluation.py) |
| Portfolio constraints, turnover, costs, and execution timing | [Backtest engine](src/llm_alpha_mining/research/backtest/engine.py) | [Backtest tests](tests/test_backtest.py) |
| Frozen candidate families and multiple-testing corrections | [Significance evaluation](src/llm_alpha_mining/research/robustness/significance.py) | [Significance tests](tests/test_significance.py) |
| Budgets, restricted feedback, persistent state, and checkpoint recovery | [Campaign orchestration](src/llm_alpha_mining/mining/orchestration) | [Campaign tests](tests/test_multigeneration_campaign.py), [recovery tests](tests/test_resumable_generation_executor.py) |

The example covers proposal review, signal computation, descriptive diagnostics, and artifact verification. Formal backtesting and campaign orchestration are tested separately. Connecting a real model and data source requires an integration runner; the example does not start a live mining campaign.

## Repository layout

```text
src/llm_alpha_mining/
  mining/       # Candidate protocols, LLM roles, DSL, state, and campaigns
  research/     # Data contracts, factors, evaluation, portfolios, and backtests
  demo/         # Reproducible example with synthetic data and scripted responses
tests/          # Tests named by behavior and research component
docs/           # Paired English and Chinese guides
```

The current tree contains one maintained implementation. Historical research runners, vendor-specific panel repair tools, unused model-training branches, caches, and submission archives are excluded. Serialized schema and operator version identifiers remain explicit so old record identities retain their meaning.

## Read further

- [Architecture and research decisions](docs/architecture.md): responsibilities, point-in-time semantics, missing data, and recovery.
- [Reproducibility and integration](docs/reproducibility.md): running the example, using a model provider, bringing your own data, and development checks.
- [Research scope and evidence](docs/project_scope.md): what this project demonstrates, what the private research established, and what remains unverified.

The source is publicly readable. An open-source license has not yet been specified; see [publication scope](docs/project_scope.md#publication-and-reuse).
