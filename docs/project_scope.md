# Research scope and evidence

**English** | [简体中文](project_scope.zh-CN.md) · [README](../README.md)

## Research contribution

This project explores LLM-assisted factor discovery as a controlled research process: structured hypotheses, deterministic computation, explicit data contracts, evaluation, and traceable artifacts. Its contribution is the implemented workflow and the treatment of research failure modes, including temporal leakage, invalid expressions, incorrect stock universes, missing observations, duplicate candidates, and interrupted execution.

The original private research included a large candidate pool, snapshot-panel correction, recomputation of dependent factors, correlation screening, and preparation of platform upload files. Those operational milestones do not establish alpha. External platform results and final out-of-sample outcomes are separate evidence; this repository makes no claim about unconfirmed returns or successful deployment.

## What a reader can verify

| Claim | Public evidence | Boundary |
| --- | --- | --- |
| Model proposals obey structured protocols and deterministic review | Role, budget, retry, and veto tests | Live model quality is not measured by scripted responses |
| Signals respect specified data and universe contracts | Factor, label, and point-in-time tests | Real input data needs separate validation |
| The packaged example is reproducible | Seeded demo, scientific signature, artifact hashes | Synthetic correlations are not performance results |
| Research components support temporal evaluation, costs, and recovery | Component tests and CI | The demo is not a full investment backtest |

## What is included

- One installable framework package with discovery and research components.
- A runnable synthetic example and functionally named tests.
- Paired English and Chinese architecture and integration guides.

Vendor-specific panel builders, historical runtime supervisors, unused model-training pipelines, local caches, actual factor catalogs, platform receipts, and submission archives are excluded. The original project is retained privately for research continuity. A public checkout is sufficient for the example, but not for reproducing private-data experiments.

## Publication and reuse

The repository is a public source release. No open-source license has been specified yet. Data and third-party platform permissions are separate from source-code permissions.

Further work can connect a real provider and data source or expand the demonstrated research path. New experiments should use new identities and output directories, retain provenance, and freeze evaluation decisions before consuming external results. Any performance claims added later should state the completed evaluation, sample period, selection procedure, costs, and limitations.
