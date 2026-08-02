"""Dataset loader tests."""

from __future__ import annotations

import json

import pytest

from autopilot.domain.entities import TrainingExample
from autopilot.domain.enums import ExampleSource, TaskType, Tier
from autopilot.domain.errors import ConfigurationError
from autopilot.infrastructure.ml.training.dataset import (
    Dataset,
    load_dataset,
    stratified_split,
    write_jsonl,
)

ROWS = [
    {"prompt": "List the days of the week.", "tier": 1, "task_type": "general", "source": "seed"},
    {"prompt": "Summarize this article.", "tier": 2, "task_type": "summarization"},
    {"prompt": "Design a distributed scheduler.", "tier": 3, "task_type": "reasoning"},
]


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def jsonl(tmp_path, rows=ROWS, name="seed.jsonl"):
    return write(tmp_path, name, "\n".join(json.dumps(r) for r in rows))


def test_jsonl_rows_become_examples(tmp_path):
    dataset = load_dataset(jsonl(tmp_path))

    assert len(dataset) == 3
    assert dataset.labels() == (Tier.SIMPLE, Tier.MODERATE, Tier.COMPLEX)
    assert dataset.examples[1].task_type is TaskType.SUMMARIZATION
    assert dataset.examples[1].source is ExampleSource.SEED


def test_blank_lines_are_ignored(tmp_path):
    path = write(tmp_path, "s.jsonl", json.dumps(ROWS[0]) + "\n\n" + json.dumps(ROWS[1]) + "\n")

    assert len(load_dataset(path)) == 2


def test_csv_rows_become_examples(tmp_path):
    path = write(
        tmp_path,
        "s.csv",
        "prompt,tier,task_type\nList the days.,1,general\nDesign a scheduler.,3,reasoning\n",
    )

    dataset = load_dataset(path)

    assert dataset.labels() == (Tier.SIMPLE, Tier.COMPLEX)


def test_yaml_rows_become_examples(tmp_path):
    path = write(
        tmp_path,
        "s.yaml",
        "examples:\n  - prompt: List the days.\n    tier: 1\n  - prompt: Prove it.\n    tier: 3\n",
    )

    assert load_dataset(path).labels() == (Tier.SIMPLE, Tier.COMPLEX)


def test_a_bare_yaml_list_is_also_accepted(tmp_path):
    path = write(tmp_path, "s.yaml", "- prompt: List the days.\n  tier: 1\n")

    assert len(load_dataset(path)) == 1


def test_json_documents_are_accepted(tmp_path):
    path = write(tmp_path, "s.json", json.dumps(ROWS))

    assert len(load_dataset(path)) == 3


def test_alternative_column_names_are_accepted(tmp_path):
    # A spreadsheet export should not need hand-editing before it loads.
    path = write(tmp_path, "s.csv", "text,label_tier\nList the days.,1\n")

    assert load_dataset(path).labels() == (Tier.SIMPLE,)


def test_files_merge_in_order(tmp_path):
    first = jsonl(tmp_path, [ROWS[0]], "a.jsonl")
    second = jsonl(tmp_path, [ROWS[2]], "b.jsonl")

    assert load_dataset(first, second).labels() == (Tier.SIMPLE, Tier.COMPLEX)


def test_repeated_prompts_at_the_same_tier_are_dropped(tmp_path):
    path = jsonl(tmp_path, [ROWS[0], dict(ROWS[0]), ROWS[1]])

    assert len(load_dataset(path)) == 2


def test_dedupe_ignores_whitespace_and_case(tmp_path):
    variant = {**ROWS[0], "prompt": "  LIST THE   DAYS of the WEEK.  "}
    path = jsonl(tmp_path, [ROWS[0], variant])

    assert len(load_dataset(path)) == 1


def test_a_relabelled_prompt_is_kept_because_it_is_a_correction(tmp_path):
    relabelled = {**ROWS[0], "tier": 3}
    path = jsonl(tmp_path, [ROWS[0], relabelled])

    assert len(load_dataset(path)) == 2


def test_dedupe_can_be_disabled(tmp_path):
    path = jsonl(tmp_path, [ROWS[0], dict(ROWS[0])])

    assert len(load_dataset(path, dedupe=False)) == 2


def test_the_fingerprint_is_stable_across_loads(tmp_path):
    path = jsonl(tmp_path)

    assert load_dataset(path).fingerprint == load_dataset(path).fingerprint


def test_the_fingerprint_changes_when_a_label_changes(tmp_path):
    original = load_dataset(jsonl(tmp_path, ROWS, "a.jsonl"))
    edited = load_dataset(jsonl(tmp_path, [{**ROWS[0], "tier": 2}, *ROWS[1:]], "b.jsonl"))

    assert original.fingerprint != edited.fingerprint


def test_tier_counts_and_balance_cover_every_tier(tmp_path):
    dataset = load_dataset(jsonl(tmp_path))

    assert dataset.tier_counts == {Tier.SIMPLE: 1, Tier.MODERATE: 1, Tier.COMPLEX: 1}
    assert sum(dataset.tier_balance.values()) == pytest.approx(1.0)


def test_an_empty_dataset_reports_zero_balance_rather_than_dividing_by_zero():
    assert Dataset(()).tier_balance == dict.fromkeys(Tier, 0.0)


def test_a_missing_prompt_is_refused(tmp_path):
    path = jsonl(tmp_path, [{"tier": 1}])

    with pytest.raises(ConfigurationError, match="missing a non-empty 'prompt'"):
        load_dataset(path)


def test_a_missing_tier_is_refused(tmp_path):
    path = jsonl(tmp_path, [{"prompt": "hi"}])

    with pytest.raises(ConfigurationError, match="missing a 'tier'"):
        load_dataset(path)


def test_an_out_of_range_tier_is_refused(tmp_path):
    path = jsonl(tmp_path, [{"prompt": "hi", "tier": 9}])

    with pytest.raises(ConfigurationError, match="invalid tier"):
        load_dataset(path)


def test_an_unknown_task_type_is_refused(tmp_path):
    path = jsonl(tmp_path, [{"prompt": "hi", "tier": 1, "task_type": "telepathy"}])

    with pytest.raises(ConfigurationError, match="unrecognized value"):
        load_dataset(path)


def test_the_error_names_the_offending_row(tmp_path):
    path = jsonl(tmp_path, [ROWS[0], ROWS[1], {"prompt": "hi"}])

    with pytest.raises(ConfigurationError, match="row 3"):
        load_dataset(path)


def test_malformed_json_names_the_line(tmp_path):
    path = write(tmp_path, "s.jsonl", json.dumps(ROWS[0]) + "\n{not json\n")

    with pytest.raises(ConfigurationError, match=":2 is not valid JSON"):
        load_dataset(path)


def test_an_unsupported_format_is_refused(tmp_path):
    path = write(tmp_path, "s.parquet", "")

    with pytest.raises(ConfigurationError, match="Unsupported dataset format"):
        load_dataset(path)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_dataset(tmp_path / "absent.jsonl")


def test_a_document_that_is_not_a_list_is_refused(tmp_path):
    path = write(tmp_path, "s.yaml", "prompt: hi\ntier: 1\n")

    with pytest.raises(ConfigurationError, match="must contain a list"):
        load_dataset(path)


def big_dataset(per_tier: int = 20) -> Dataset:
    return Dataset(
        tuple(
            TrainingExample(prompt=f"prompt {tier} number {i}", tier=tier)
            for tier in Tier
            for i in range(per_tier)
        )
    )


def test_the_split_preserves_tier_balance():
    train, holdout = stratified_split(big_dataset(), holdout_fraction=0.2)

    assert holdout.tier_counts == {Tier.SIMPLE: 4, Tier.MODERATE: 4, Tier.COMPLEX: 4}
    assert train.tier_counts == {Tier.SIMPLE: 16, Tier.MODERATE: 16, Tier.COMPLEX: 16}


def test_the_split_is_a_partition_with_no_overlap():
    dataset = big_dataset()
    train, holdout = stratified_split(dataset)

    train_hashes = {e.prompt_hash for e in train}
    holdout_hashes = {e.prompt_hash for e in holdout}
    assert not train_hashes & holdout_hashes
    assert len(train_hashes | holdout_hashes) == len(dataset)


def test_the_split_flags_holdout_rows_so_the_flag_survives_a_round_trip():
    train, holdout = stratified_split(big_dataset())

    assert all(e.is_holdout for e in holdout)
    assert not any(e.is_holdout for e in train)


def test_the_split_is_deterministic_for_a_given_seed():
    dataset = big_dataset()

    first = stratified_split(dataset, seed=7)[1].prompts()
    second = stratified_split(dataset, seed=7)[1].prompts()

    assert first == second


def test_a_different_seed_selects_different_rows():
    dataset = big_dataset()

    assert stratified_split(dataset, seed=1)[1].prompts() != stratified_split(dataset, seed=2)[
        1
    ].prompts()


def test_a_tiny_tier_still_contributes_a_holdout_row():
    # round() rather than truncation: 5 rows at 20% must reserve 1, not 0.
    dataset = Dataset(tuple(TrainingExample(prompt=f"p{i}", tier=Tier.COMPLEX) for i in range(5)))

    _, holdout = stratified_split(dataset, holdout_fraction=0.2)

    assert len(holdout) == 1


def test_an_impossible_holdout_fraction_is_refused():
    with pytest.raises(ConfigurationError, match="between 0 and 1"):
        stratified_split(big_dataset(), holdout_fraction=1.0)


def test_the_filter_selects_holdout_rows():
    train, holdout = stratified_split(big_dataset())
    combined = Dataset(train.examples + holdout.examples)

    assert len(combined.filter(holdout=True)) == len(holdout)
    assert len(combined.filter(holdout=False)) == len(train)
    assert len(combined.filter()) == len(combined)


def test_written_jsonl_reloads_identically(tmp_path):
    _, holdout = stratified_split(big_dataset())
    path = tmp_path / "nested" / "holdout.jsonl"

    write_jsonl(holdout, path)
    reloaded = load_dataset(path)

    assert reloaded.prompts() == holdout.prompts()
    assert reloaded.labels() == holdout.labels()
    assert all(e.is_holdout for e in reloaded)
    assert reloaded.fingerprint == holdout.fingerprint


def test_a_false_holdout_string_from_csv_is_not_truthy(tmp_path):
    # "false" is a non-empty string; naive coercion would leak it into holdout.
    path = write(tmp_path, "s.csv", "prompt,tier,is_holdout\nList the days.,1,false\n")

    assert load_dataset(path).examples[0].is_holdout is False
