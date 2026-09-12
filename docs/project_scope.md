# Project scope and status

**English** | [简体中文](project_scope.zh-CN.md) · [Back to README](../README.md)

This is a compact public edition of an LLM-assisted automated factor research framework. It separates the reusable framework from the outputs of one research campaign. The framework includes candidate proposal and review, data and factor contracts, computation, temporal validation, portfolio and cost evaluation, and recoverable orchestration. Vendor configuration, actual factor catalogs, and platform records remain in the private research archive.

## What this repository demonstrates

- Actual framework components and their portable synthetic tests.
- An offline workflow from structured candidates to signal files, descriptive diagnostics, and SHA verification.
- Engineering constraints for missing values, point-in-time stock universes, risk vetoes, budgets and retries, lineage, and checkpoint recovery.
- Documentation of bulk recomputation and traceable research delivery in the original project.

## Research status

Local recomputation and preparation of upload files are complete. External platform retesting and the current batch's final delivery await feedback from the project owner. The public edition does not include unconfirmed out-of-sample performance, and candidate counts are not evidence of effective factors. Closing this research phase does not automatically start a new mining campaign.

Future work can build on the existing interfaces to connect data and models or improve search and execution efficiency. Resumed research should use a new experiment identity, retain necessary lineage and contracts, and revalidate the runtime environment.

## Publication

[ZebinZ/llm-alpha-mining](https://github.com/ZebinZ/llm-alpha-mining) publishes framework source code, a synthetic example, and tests. Framework publication proceeds independently of platform retesting and final out-of-sample (OOS) delivery. Any research results added later must reflect completed verification.

No open-source license has been specified yet. Market data, the specific factor pool, platform records, and private research archives are kept locally and are outside this repository.
