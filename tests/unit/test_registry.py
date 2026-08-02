from __future__ import annotations

import pytest

from autopilot.config import PROJECT_ROOT
from autopilot.domain.enums import Provider, Tier
from autopilot.domain.errors import ConfigurationError, ModelNotFoundError, NoCapableModelError
from autopilot.infrastructure.providers.registry import YamlModelRegistry
from tests.conftest import make_model


def test_loads_the_shipped_models_yaml():
    registry = YamlModelRegistry.from_yaml(PROJECT_ROOT / "configs" / "models.yaml")

    assert len(registry.all()) >= 6
    assert registry.get("anthropic:claude-opus-5").tier is Tier.COMPLEX
    assert registry.get("ollama:llama3.1-8b").is_free


def test_shipped_yaml_has_no_date_suffixed_aliases():
    registry = YamlModelRegistry.from_yaml(PROJECT_ROOT / "configs" / "models.yaml")
    for model in registry.all():
        if model.provider is Provider.ANTHROPIC:
            assert not model.vendor_model_id[-8:].isdigit(), model.vendor_model_id


def test_every_tier_has_at_least_one_shipped_model():
    registry = YamlModelRegistry.from_yaml(PROJECT_ROOT / "configs" / "models.yaml")
    for tier in Tier:
        assert registry.by_tier(tier), f"tier {int(tier)} has no active model"


def test_by_tier_orders_cheapest_first(registry: YamlModelRegistry):
    ids = [m.id for m in registry.by_tier(Tier.MODERATE)]
    assert ids == ["openai:gpt-4o-mini", "anthropic:claude-sonnet-5"]


def test_cheapest_returns_the_free_local_model(registry: YamlModelRegistry):
    assert registry.cheapest(Tier.SIMPLE).id == "ollama:llama3.1-8b"


def test_by_tier_excludes_inactive_models():
    registry = YamlModelRegistry(
        [
            make_model("openai:gpt-4o-mini", tier=Tier.SIMPLE, active=False),
            make_model("openai:gpt-4o", tier=Tier.SIMPLE, input_cost="2.50",
                       output_cost="10.00"),
        ]
    )
    assert [m.id for m in registry.by_tier(Tier.SIMPLE)] == ["openai:gpt-4o"]


def test_get_raises_for_unknown_model(registry: YamlModelRegistry):
    with pytest.raises(ModelNotFoundError, match="nope:model"):
        registry.get("nope:model")


def test_cheapest_raises_when_tier_is_empty():
    registry = YamlModelRegistry([make_model("openai:gpt-4o-mini", tier=Tier.SIMPLE)])
    with pytest.raises(NoCapableModelError):
        registry.cheapest(Tier.COMPLEX)


def test_empty_registry_is_rejected():
    with pytest.raises(ConfigurationError, match="empty"):
        YamlModelRegistry([])


def test_duplicate_ids_are_rejected():
    with pytest.raises(ConfigurationError, match="Duplicate"):
        YamlModelRegistry([make_model("openai:gpt-4o-mini"), make_model("openai:gpt-4o-mini")])


def test_missing_file_raises_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        YamlModelRegistry.from_yaml(tmp_path / "absent.yaml")


def test_invalid_yaml_raises_configuration_error(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        YamlModelRegistry.from_yaml(path)


def test_document_without_models_list_is_rejected():
    with pytest.raises(ConfigurationError, match="'models' list"):
        YamlModelRegistry.from_dict({"version": 1})


def test_entry_with_negative_price_is_rejected():
    raw = {
        "version": 1,
        "models": [
            {
                "id": "openai:bad",
                "provider": "openai",
                "vendor_model_id": "bad",
                "tier": 1,
                "input_cost_per_mtok": "-1",
                "output_cost_per_mtok": "0",
                "max_context_tokens": 1000,
                "max_output_tokens": 100,
                "expected_latency_ms": 10,
            }
        ],
    }
    with pytest.raises(ConfigurationError, match="openai:bad"):
        YamlModelRegistry.from_dict(raw)


def test_entry_with_unknown_provider_is_rejected():
    raw = {
        "models": [
            {
                "id": "mystery:model",
                "provider": "mystery",
                "vendor_model_id": "m",
                "tier": 1,
                "input_cost_per_mtok": "1",
                "output_cost_per_mtok": "1",
                "max_context_tokens": 1000,
                "max_output_tokens": 100,
                "expected_latency_ms": 10,
            }
        ]
    }
    with pytest.raises(ConfigurationError, match="mystery:model"):
        YamlModelRegistry.from_dict(raw)


def test_fits_context_guards_window_and_output_cap(registry: YamlModelRegistry):
    model = registry.get("openai:gpt-4o-mini")
    assert registry.fits_context(model, prompt_tokens=1_000, max_tokens=1_000)
    assert not registry.fits_context(model, prompt_tokens=200_000, max_tokens=100)
    assert not registry.fits_context(model, prompt_tokens=10, max_tokens=99_999)


def test_map_version_tracks_config_version():
    registry = YamlModelRegistry([make_model()], version=7)
    assert registry.map_version == "models-v7"
