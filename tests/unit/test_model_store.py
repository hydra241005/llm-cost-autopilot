"""Tests for the filesystem-backed classifier artifact store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from autopilot.domain.entities import ClassifierMetadata
from autopilot.domain.enums import ClassifierStatus
from autopilot.domain.errors import ClassifierError
from autopilot.infrastructure.ml.model_store import FilesystemClassifierStore


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = FilesystemClassifierStore(tmp_path)
    metadata = ClassifierMetadata(
        version="v1",
        status=ClassifierStatus.CANDIDATE,
        algorithm="logistic_regression",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_hash="abc123",
        dataset_size=10,
        feature_schema_version="fx-1",
    )

    store.save("v1", artifact={"model": "ok"}, metadata=metadata)

    artifact, reloaded = store.load("v1")

    assert artifact == {"model": "ok"}
    assert reloaded.version == "v1"
    assert reloaded.status is ClassifierStatus.CANDIDATE


def test_promote_switches_the_active_version(tmp_path: Path) -> None:
    store = FilesystemClassifierStore(tmp_path)
    first = ClassifierMetadata(
        version="v1",
        status=ClassifierStatus.CANDIDATE,
        algorithm="logistic_regression",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_hash="abc123",
        dataset_size=10,
        feature_schema_version="fx-1",
    )
    second = first.model_copy(update={"version": "v2", "status": ClassifierStatus.CANDIDATE})

    store.save("v1", artifact={"a": 1}, metadata=first)
    store.save("v2", artifact={"b": 2}, metadata=second)
    store.promote("v2")

    active_artifact, active_meta = store.load_active()
    assert active_artifact == {"b": 2}
    assert active_meta is not None
    assert active_meta.version == "v2"
    assert active_meta.status is ClassifierStatus.ACTIVE


def test_load_missing_version_raises_classifier_error(tmp_path: Path) -> None:
    store = FilesystemClassifierStore(tmp_path)

    with pytest.raises(ClassifierError, match="No such version"):
        store.load("missing")


def test_list_versions_returns_newest_first(tmp_path: Path) -> None:
    store = FilesystemClassifierStore(tmp_path)
    first = ClassifierMetadata(
        version="v1",
        status=ClassifierStatus.CANDIDATE,
        algorithm="logistic_regression",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_hash="abc123",
        dataset_size=10,
        feature_schema_version="fx-1",
    )
    second = first.model_copy(update={"version": "v2"})

    store.save("v1", artifact={"a": 1}, metadata=first)
    store.save("v2", artifact={"b": 2}, metadata=second)

    assert [meta.version for meta in store.list_versions()] == ["v2", "v1"]
