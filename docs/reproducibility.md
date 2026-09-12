# Reproducibility and integration

**English** | [简体中文](reproducibility.zh-CN.md) · [README](../README.md)

## Run and verify

Follow the installation commands in the [README](../README.md). The project uses a `src/` layout: install it before importing `llm_alpha_mining`, including when running from the repository root.

```bash
alpha-demo --output outputs/run-a
alpha-demo --output outputs/run-b
alpha-demo --output outputs/run-a --verify
```

With the same software environment and seed, the two `report.json` files should have matching `scientific_signature` values. The call ledger includes timestamps, so entire output directories need not match byte for byte. Verification rejects altered or missing artifacts; it is not a statistical validity test.

After installation, the equivalent module command is `python -m llm_alpha_mining.demo.run`. Use `alpha-demo --help` for options.

## Inspect a research workspace

The workspace CLI creates frozen protocol and artifact identities with a persistent run registry:

```bash
alpha-workspace init --workspace outputs/research-workspace
alpha-workspace status --workspace outputs/research-workspace
alpha-workspace verify --workspace outputs/research-workspace
```

These commands initialize and inspect a workspace. They do not start a campaign. Programmatic orchestration examples are in [campaign tests](../tests/test_multigeneration_campaign.py) and [resumable execution tests](../tests/test_resumable_generation_executor.py).

## Connect a real model

The demo explicitly uses `FakeTransport`. A live runner can construct `LiveProviderConfig` and `LiveStructuredTransport`, then inject the transport into `StructuredCallExecutor`. See the [transport implementation](../src/llm_alpha_mining/research/agents/live_transport.py) and [offline transport tests](../tests/test_live_llm_transport.py) for complete configuration examples.

Configuration includes an HTTPS host allowlist, credential environment-variable names, model identifier, policy bindings, response limits, and cost limits. Confirm your provider's API compatibility, model availability, and pricing before integration. Demo model names and billing are fictional. Credentials belong in environment variables, not candidate definitions or logs. Live provider compatibility is not exercised by the offline CI suite.

## Bring your own data

1. Define schema, frequency, availability rules, and an immutable snapshot.
2. Build `DataBatch` objects from a source you are authorized to use.
3. Supply instrument identity, status changes, the signal-time universe, and availability times.
4. Bind accepted candidates using `factor_spec_from_candidate`; preserve missing values and warm-up windows.
5. Freeze labels, temporal splits, costs, and selection rules before formal evaluation.
6. Export values and manifests; keep independent evaluation separate from the search loop.

The [demo runner](../src/llm_alpha_mining/demo/run.py) shows the data-to-factor path. [Factor contract tests](../tests/test_factor_contracts.py) illustrate invalid and valid bindings. Completeness declarations in synthetic fixtures do not certify a real vendor's data.

## Development and packaging

For source edits, install with `python -m pip install -e '.[test]'`, then run `python -m pytest -q`. For a release check, build and install the wheel instead of relying on an editable checkout:

```bash
python -m pip wheel --no-deps . --wheel-dir dist
python -m pip install --force-reinstall --no-deps dist/llm_alpha_mining-0.2.0-py3-none-any.whl
python -m pytest -q
```

CI builds and installs a wheel on Python 3.12 and 3.13, runs the tests, runs the demo from outside the checkout, verifies its files, and checks workspace initialization. See [Actions](https://github.com/ZebinZ/llm-alpha-mining/actions/workflows/tests.yml) for the current result.

`requirements-validated.txt` records direct dependency versions used for local acceptance; it is not a full transitive lockfile. The project declares compatible dependency ranges. Record your environment and seed when comparing scientific signatures.

The public edition uses `llm_alpha_mining.mining` and `llm_alpha_mining.research`. Previous package paths are not compatibility aliases. A serialized schema version may still contain historical identifiers, as explained in [architecture](architecture.md#package-and-artifact-identity).
