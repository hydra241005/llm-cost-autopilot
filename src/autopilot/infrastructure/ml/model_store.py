"""Filesystem-backed persistence for trained classifier artifacts.

The router can hot-reload a promoted artifact without changing its interface or
its dependency injection shape. The implementation is intentionally simple: it
stores one artifact payload and one metadata document per version under a
versioned directory and keeps a small pointer file that records the currently
active version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from autopilot.domain.entities import ClassifierMetadata
from autopilot.domain.enums import ClassifierStatus
from autopilot.domain.errors import ClassifierError
from autopilot.domain.interfaces import ClassifierStore


class FilesystemClassifierStore(ClassifierStore):
    """Persist classifier artifacts and metadata on the local filesystem."""

    def __init__(self, root: str | Path | None = None) -> None:
        """Create the store under ``root``.

        Args:
            root: Directory for all versions. Defaults to a local ``artifacts/classifier``
                directory under the project root when omitted.
        """
        self._root = Path(root or "artifacts/classifier")
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, version: str, artifact: Any, metadata: ClassifierMetadata) -> None:
        """Persist ``artifact`` and ``metadata`` under ``version``."""
        version_dir = self._root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, version_dir / "artifact.joblib")
        (version_dir / "metadata.json").write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, version: str) -> tuple[Any, ClassifierMetadata]:
        """Return the artifact and metadata stored under ``version``."""
        version_dir = self._root / version
        artifact_path = version_dir / "artifact.joblib"
        metadata_path = version_dir / "metadata.json"
        if not artifact_path.exists() or not metadata_path.exists():
            raise ClassifierError(f"No such version {version!r} in classifier store")

        try:
            artifact = joblib.load(artifact_path)
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ClassifierError(f"Classifier artifact {version!r} is unreadable: {exc}") from exc

        try:
            metadata = ClassifierMetadata.model_validate(payload)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ClassifierError(f"Classifier metadata {version!r} is invalid: {exc}") from exc
        return artifact, metadata

    @property
    def active_version(self) -> str | None:
        """Return the currently promoted version, if one exists."""
        pointer_path = self._root / "active_version"
        if not pointer_path.exists():
            return None

        version = pointer_path.read_text(encoding="utf-8").strip()
        return version or None

    def load_active(self) -> tuple[Any, ClassifierMetadata] | None:
        """Return the currently active artifact, or ``None`` when none is promoted."""
        version = self.active_version
        if version is None:
            return None
        return self.load(version)

    def list_versions(self) -> list[ClassifierMetadata]:
        """Return metadata for every stored version, newest first."""
        versions = sorted(
            [p.name for p in self._root.iterdir() if p.is_dir() and p.name != "active_version"],
            key=lambda name: name,
            reverse=True,
        )
        return [self.load(version)[1] for version in versions]

    def promote(self, version: str) -> None:
        """Mark ``version`` active and archive the previously promoted version."""
        if not (self._root / version).exists():
            raise ClassifierError(f"Cannot promote unknown version {version!r}")

        artifact, metadata = self.load(version)
        updated = metadata.model_copy(update={"status": ClassifierStatus.PRODUCTION})
        self.save(version, artifact, updated)

        previous_version = self.active_version
        if previous_version and previous_version != version:
            previous_artifact, previous_metadata = self.load(previous_version)
            archived = previous_metadata.model_copy(update={"status": ClassifierStatus.ARCHIVED})
            self.save(previous_version, previous_artifact, archived)

        (self._root / "active_version").write_text(version, encoding="utf-8")
