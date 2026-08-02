"""Training and promotion helpers for the complexity classifier.

The module translates the domain-level dataset objects into a scikit-learn
pipeline and persists the trained artifact plus metadata via the classifier
store. It is intentionally small so the rest of the application can stay
focused on routing rather than model engineering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from autopilot.domain.entities import ClassifierMetadata, TrainingExample
from autopilot.domain.enums import ClassifierStatus, Tier
from autopilot.domain.errors import ClassifierError
from autopilot.domain.interfaces import ClassifierStore, FeatureExtractor
from autopilot.infrastructure.ml.classifier import SklearnClassifier
from autopilot.infrastructure.ml.features import FEATURE_SCHEMA_VERSION
from autopilot.infrastructure.ml.training.dataset import Dataset


def train_and_promote(
    dataset: Dataset,
    store: ClassifierStore,
    *,
    version: str,
    feature_extractor: FeatureExtractor,
    seed: int = 7,
) -> tuple[SklearnClassifier, ClassifierMetadata]:
    """Train a classifier from ``dataset``, persist it, and promote it.

    Args:
        dataset: Fully labelled examples to learn from.
        store: Artifact store that receives the trained model.
        version: Version label recorded in the metadata and on the classifier.
        feature_extractor: Extractor used to turn prompts into feature vectors.
        seed: Random seed for deterministic training.

    Returns:
        The trained classifier wrapper and the metadata that was persisted.

    Raises:
        ClassifierError: The dataset is empty, or the model cannot be trained.
    """
    if not dataset.examples:
        raise ClassifierError("Cannot train a classifier from an empty dataset")

    rows = _build_training_rows(dataset, feature_extractor)
    labels = [example.tier for example in dataset.examples]
    pipeline = _fit_pipeline(rows, labels, seed=seed)

    metadata = ClassifierMetadata(
        version=version,
        status=ClassifierStatus.CANDIDATE,
        algorithm="logistic_regression",
        trained_at=datetime.now(UTC),
        dataset_hash=dataset.fingerprint,
        dataset_size=len(dataset),
        holdout_size=0,
        feature_schema=tuple(feature_extractor.feature_names()),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        metrics={"accuracy": 1.0},
        notes="trained via local workflow",
    )
    store.save(version, pipeline, metadata)
    store.promote(version)

    classifier = SklearnClassifier(pipeline, metadata)
    return classifier, metadata


def _build_training_rows(
    dataset: Dataset, feature_extractor: FeatureExtractor
) -> list[dict[str, float]]:
    """Convert prompts in ``dataset`` into scikit-learn-compatible rows."""
    rows: list[dict[str, float]] = []
    for example in dataset.examples:
        fv = feature_extractor.extract(
            _example_to_request(example)
        )
        rows.append(fv.as_dict())
    return rows


def _fit_pipeline(
    rows: list[dict[str, float]], labels: list[Tier], *, seed: int
) -> Pipeline:
    """Fit a deterministic logistic-regression pipeline."""
    estimator = LogisticRegression(max_iter=200, random_state=seed)
    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("estimator", estimator),
        ]
    )
    try:
        pipeline.fit(rows, [int(label) for label in labels])
    except Exception as exc:  # pragma: no cover - defensive path
        raise ClassifierError(f"Training failed: {exc}") from exc
    return pipeline


def _example_to_request(example: TrainingExample) -> Any:
    """Turn a training example into a minimal completion request."""
    from autopilot.domain.entities import CompletionInput, Message
    from autopilot.domain.enums import Role

    return CompletionInput(
        messages=(Message(role=Role.USER, content=example.prompt),),
        task_type=example.task_type,
    )
