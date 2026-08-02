"""Operational lifecycle helpers for classifier artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from autopilot.domain.entities import ClassifierMetadata, TrainingJob
from autopilot.domain.enums import ClassifierStatus, TrainingJobStatus
from autopilot.domain.errors import ClassifierError
from autopilot.infrastructure.ml.model_store import FilesystemClassifierStore


class ClassifierLifecycleService:
    """Manage promotion, rollback, and job-tracking for classifier versions."""

    def __init__(self, store: FilesystemClassifierStore) -> None:
        self._store = store
        self._jobs: dict[str, TrainingJob] = {}

    def transition_to(self, version: str, status: ClassifierStatus) -> ClassifierMetadata:
        """Update a version's lifecycle state."""
        artifact, metadata = self._store.load(version)
        if status is ClassifierStatus.PRODUCTION and metadata.status is ClassifierStatus.ARCHIVED:
            raise ClassifierError("Archived classifiers cannot be re-promoted")

        updated = metadata.model_copy(update={"status": status})
        self._store.save(version, artifact, updated)
        return updated

    def promote(self, version: str) -> ClassifierMetadata:
        """Promote a version to production and archive the previous production version."""
        self._store.promote(version)
        _, metadata = self._store.load(version)
        return metadata

    def rollback(self, version: str) -> ClassifierMetadata:
        """Rollback to a previous version by promoting it and archiving the current one."""
        self._store.promote(version)
        _, metadata = self._store.load(version)
        return metadata

    def start_job(self, version: str, *, job_id: str | None = None) -> TrainingJob:
        """Create a training job record for ``version``."""
        job_key = job_id or f"job-{len(self._jobs)+1}"
        job = TrainingJob(
            job_id=job_key,
            version=version,
            status=TrainingJobStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        self._jobs[job_key] = job
        return job

    def mark_running(self, job_id: str) -> TrainingJob:
        """Mark a training job as running."""
        job = self._jobs[job_id]
        updated = job.model_copy(update={"status": TrainingJobStatus.RUNNING, "started_at": datetime.now(UTC)})
        self._jobs[job_id] = updated
        return updated

    def mark_succeeded(self, job_id: str, *, artifact_version: str) -> TrainingJob:
        """Mark a training job as succeeded and record the produced artifact version."""
        job = self._jobs[job_id]
        updated = job.model_copy(
            update={
                "status": TrainingJobStatus.SUCCEEDED,
                "finished_at": datetime.now(UTC),
                "artifact_version": artifact_version,
            }
        )
        self._jobs[job_id] = updated
        return updated

    def mark_failed(self, job_id: str, *, error: str) -> TrainingJob:
        """Mark a training job as failed."""
        job = self._jobs[job_id]
        updated = job.model_copy(
            update={"status": TrainingJobStatus.FAILED, "finished_at": datetime.now(UTC), "error": error}
        )
        self._jobs[job_id] = updated
        return updated

    def get_job(self, job_id: str) -> TrainingJob:
        """Return the current state for a training job."""
        return self._jobs[job_id]
