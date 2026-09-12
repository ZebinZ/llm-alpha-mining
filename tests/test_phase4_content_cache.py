from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from alpha_research.experiments import ResearchContentCache, SignalRealizationKey


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _key(*, partitions: tuple[str, ...] = ("2020",), parent: str | None = None):
    return SignalRealizationKey(
        factor_definition_hash=_hash("factor"),
        data_snapshot_hash=_hash("data"),
        security_contract_hash=_hash("security"),
        availability_contract_hash=_hash("availability"),
        frequency_spec_hash=_hash("frequency"),
        transform_spec_hash=_hash("transform"),
        code_snapshot_hash=_hash("code"),
        environment_hash=_hash("environment"),
        partition_hashes=tuple(_hash(value) for value in partitions),
        parent_cache_key_hash=parent,
    )


def test_content_cache_records_cold_miss_and_verified_warm_hit(tmp_path) -> None:
    key = _key()
    with ResearchContentCache(tmp_path / "cache", maximum_bytes=128) as cache:
        assert cache.get(key) is None
        entry = cache.put(key, b"signal-realization")
        assert entry.cache_key_hash == key.content_hash
        assert cache.get(key) == b"signal-realization"
        assert cache.stats() == {
            "entry_count": 1,
            "bytes_used": 18,
            "hit_count": 1,
            "miss_count": 1,
            "hit_rate": 0.5,
        }
        assert cache.verify_integrity() == ()


def test_cache_key_is_immutable_and_all_identity_fields_participate(tmp_path) -> None:
    key = _key()
    changed_code = replace(key, code_snapshot_hash=_hash("code-v2"))
    changed_data = replace(key, data_snapshot_hash=_hash("data-v2"))
    assert len({key.content_hash, changed_code.content_hash, changed_data.content_hash}) == 3

    with ResearchContentCache(tmp_path / "cache", maximum_bytes=128) as cache:
        first = cache.put(key, b"same")
        second = cache.put(key, b"same")
        assert first.artifact_hash == second.artifact_hash
        with pytest.raises(ValueError, match="different content"):
            cache.put(key, b"different")


def test_incremental_cache_key_binds_parent_and_new_partition(tmp_path) -> None:
    base = _key(partitions=("2020",))
    incremental = _key(partitions=("2021",), parent=base.content_hash)
    independent = _key(partitions=("2021",))
    assert incremental.content_hash != independent.content_hash

    with ResearchContentCache(tmp_path / "cache", maximum_bytes=128) as cache:
        cache.put(base, b"2020-realization")
        cache.put(incremental, b"2021-increment")
        assert cache.get(base) == b"2020-realization"
        assert cache.get(incremental) == b"2021-increment"


def test_cache_capacity_is_enforced_without_evicting_immutable_entries(tmp_path) -> None:
    with ResearchContentCache(tmp_path / "cache", maximum_bytes=4) as cache:
        cache.put(_key(), b"1234")
        with pytest.raises(ValueError, match="byte budget"):
            cache.put(_key(partitions=("2021",)), b"5")
        assert cache.stats()["entry_count"] == 1


def test_cache_configuration_is_persistently_bound(tmp_path) -> None:
    root = tmp_path / "cache"
    with ResearchContentCache(root, maximum_bytes=4):
        pass
    with pytest.raises(ValueError, match="binding differs"):
        ResearchContentCache(root, maximum_bytes=5)
