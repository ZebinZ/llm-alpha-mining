# Reproducibility and integration

**English** | [简体中文](reproducibility.zh-CN.md) · [Back to README](../README.md)

## Reproduce the offline demo

The [README](../README.md) contains the maintained installation and run commands. With the same software environment and random seed, two runs into new output directories should produce the same `scientific_signature` in `report.json`. The call ledger includes timestamps, so the complete directories are not expected to be identical byte for byte.

`requirements-validated.txt` records the direct dependency versions used for local acceptance. It is not a complete lockfile with all transitive dependencies and hashes. CI uses Python 3.12 and 3.13 with the compatible dependency ranges declared by the project. The published version passed 251 tests, the demo, and hash verification on each Python version; check [GitHub Actions](https://github.com/ZebinZ/llm-alpha-mining/actions/workflows/tests.yml) for the latest status.

You can also run the module directly without installing the command-line entry point, provided the required dependencies are available and you are in the repository root:

```bash
python -m alpha_demo.run --output outputs/another-demo
python -m alpha_demo.run --output outputs/another-demo --verify
```

## Connect a real model

The offline entry point explicitly uses `FakeTransport`. For live calls, construct `LiveProviderConfig` and `LiveStructuredTransport` in your own runner, then pass the transport to `StructuredCallExecutor`. Configuration includes an HTTPS host allowlist, environment-variable names, a provider model identifier, and limits on cost and response size. See [`alpha_research/agents/live_transport.py`](../alpha_research/agents/live_transport.py) for the actual fields.

Confirm API compatibility, available models, and current pricing when integrating a provider. The demo's fictional model identifiers and simulated billing are not live-service configuration. Supply credentials through environment variables; do not put keys in candidate definitions, logs, or the repository.

## Connect research data

1. Define the data schema, frequency, availability rules, and an immutable snapshot.
2. Implement an adapter or construct `DataBatch` objects from a data source you are authorized to use.
3. Supply instrument identity, status changes, trading-universe information, and data availability times.
4. Bind candidates to the data and operator registry before computing them; preserve warm-up periods and missing values.
5. Freeze labels, temporal validation, costs, and selection rules before formal evaluation.
6. Export artifacts and hashes, then determine final deliverables after independent evaluation.

Completeness assertions for the demo's fictional data apply only to the generated fixtures. They are not evidence that a real vendor's data has been validated.

## Restore the original research

The public repository reproduces framework behavior and the demo. It cannot reproduce every private-data result on its own. Restoring the original research additionally requires the frozen catalogs, data contracts, panels, formulas, run receipts, and version notes in the private archive, with their hashes verified. New experiments should write to new output directories rather than overwrite frozen results.
