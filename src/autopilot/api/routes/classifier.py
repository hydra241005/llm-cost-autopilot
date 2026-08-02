"""Administrative classifier lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from autopilot.domain.entities import ClassifierMetadata
from autopilot.domain.enums import ClassifierStatus, TrainingJobStatus
from autopilot.domain.errors import ClassifierError
from autopilot.infrastructure.ml.training.lifecycle import ClassifierLifecycleService

router = APIRouter(prefix="/v1/classifiers", tags=["classifiers"])


class TransitionRequest(BaseModel):
    """Request body for changing a classifier's lifecycle state."""

    status: ClassifierStatus = Field(...)


class PromoteRequest(BaseModel):
    """Request body for promoting a classifier version."""

    version: str = Field(..., min_length=1)


class TrainingJobResponse(BaseModel):
    """Public view of an asynchronous training job."""

    job_id: str
    version: str
    status: TrainingJobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    artifact_version: str | None = None


@router.get("", response_model=list[ClassifierMetadata], summary="List classifier versions")
async def list_classifiers(request: Request) -> list[ClassifierMetadata]:
    """List persisted classifier versions and their metadata."""
    store = request.app.state.classifier_store
    return store.list_versions()


@router.post("/{version}/transition", response_model=ClassifierMetadata)
async def transition_classifier(version: str, payload: TransitionRequest, request: Request) -> ClassifierMetadata:
    """Advance or retreat a classifier version through the lifecycle states."""
    service = _service(request)
    try:
        return service.transition_to(version, payload.status)
    except ClassifierError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{version}/promote", response_model=ClassifierMetadata)
async def promote_classifier(version: str, request: Request) -> ClassifierMetadata:
    """Promote a classifier version to production."""
    service = _service(request)
    try:
        return service.promote(version)
    except ClassifierError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{version}/rollback", response_model=ClassifierMetadata)
async def rollback_classifier(version: str, request: Request) -> ClassifierMetadata:
    """Rollback to a previous classifier version."""
    service = _service(request)
    try:
        return service.rollback(version)
    except ClassifierError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training_job(payload: PromoteRequest, request: Request) -> TrainingJobResponse:
    """Create an asynchronous training job for a new classifier version."""
    service = _service(request)
    job = service.start_job(payload.version)
    return TrainingJobResponse(
        job_id=job.job_id,
        version=job.version,
        status=job.status,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        error=job.error,
        artifact_version=job.artifact_version,
    )


def _service(request: Request) -> ClassifierLifecycleService:
    if not hasattr(request.app.state, "classifier_lifecycle"):
        request.app.state.classifier_lifecycle = ClassifierLifecycleService(request.app.state.classifier_store)
    return request.app.state.classifier_lifecycle
