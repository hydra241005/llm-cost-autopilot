"""Tests for the classifier training and promotion workflow."""

from __future__ import annotations

from pathlib import Path

from autopilot.domain.entities import CompletionInput, Message, TrainingExample
from autopilot.domain.enums import Role, Tier
from autopilot.infrastructure.ml.classifier import SklearnClassifier
from autopilot.infrastructure.ml.features import PromptFeatureExtractor
from autopilot.infrastructure.ml.model_store import FilesystemClassifierStore
from autopilot.infrastructure.ml.training.dataset import Dataset
from autopilot.infrastructure.ml.training.train import train_and_promote


def test_train_and_promote_persists_a_trained_classifier(tmp_path: Path) -> None:
    extractor = PromptFeatureExtractor()
    examples = [
        TrainingExample(prompt="List the days of the week.", tier=Tier.SIMPLE),
        TrainingExample(prompt="Summarize this document in one paragraph.", tier=Tier.SIMPLE),
        TrainingExample(prompt="Explain the steps to repair the server.", tier=Tier.MODERATE),
        TrainingExample(prompt="Compare two architectures and explain the trade-offs.", tier=Tier.MODERATE),
        TrainingExample(prompt="Analyze this design, justify the choice, and then propose upgrades.", tier=Tier.COMPLEX),
        TrainingExample(prompt="Evaluate the system, compare alternatives, and explain why the best option works.", tier=Tier.COMPLEX),
    ]
    dataset = Dataset(tuple(examples))
    store = FilesystemClassifierStore(tmp_path)

    classifier, metadata = train_and_promote(
        dataset,
        store,
        version="v1",
        feature_extractor=extractor,
        seed=7,
    )

    assert metadata.version == "v1"
    assert classifier.version == "v1"

    artifact, reloaded = store.load("v1")
    assert artifact is not None
    assert reloaded.version == "v1"

    loaded_classifier = SklearnClassifier(artifact, reloaded)
    prediction = loaded_classifier.predict(
        extractor.extract(
            CompletionInput(messages=(Message(role=Role.USER, content="Analyze and compare these designs."),))
        )
    )
    assert prediction.classifier_version == "v1"
    assert prediction.tier in {Tier.SIMPLE, Tier.MODERATE, Tier.COMPLEX}

    active_artifact, active_metadata = store.load_active()
    assert active_artifact is not None
    assert active_metadata is not None
    assert active_metadata.version == "v1"
