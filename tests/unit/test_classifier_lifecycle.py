"""Tests for classifier lifecycle transitions and rollback support."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from autopilot.domain.entities import ClassifierMetadata
from autopilot.domain.enums import ClassifierStatus
from autopilot.infrastructure.ml.model_store import FilesystemClassifierStore
from autopilot.infrastructure.ml.training.lifecycle import ClassifierLifecycleService


def test_state_machine_transitions_and_rollback(tmp_path: Path) -> None:
    store = FilesystemClassifierStore(tmp_path)
    service = ClassifierLifecycleService(store)

    first = ClassifierMetadata(
        version="v1",
        status=ClassifierStatus.DRAFT,
        algorithm="logistic_regression",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_hash="abc",
        dataset_size=10,
        feature_schema_version="fx-1",
    )
    second = first.model_copy(update={"version": "v2"})

    store.save("v1", artifact={"one": 1}, metadata=first)
    store.save("v2", artifact={"two": 2}, metadata=second)

    service.transition_to("v1", ClassifierStatus.VALIDATED)
    service.transition_to("v1", ClassifierStatus.CANDIDATE)
    service.promote("v1")

    active_version = store.active_version
    assert active_version == "v1"
    metadata = store.load("v1")[1]
    assert metadata.status is ClassifierStatus.PRODUCTION

    service.rollback("v2")
    assert store.active_version == "v2"
    assert store.load("v2")[1].status is ClassifierStatus.PRODUCTION
