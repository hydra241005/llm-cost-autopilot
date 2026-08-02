"""Training dataset loading, deduplication, and splitting.

Reads labelled prompts from CSV, JSONL, or YAML — three formats because the
seed set is hand-written (JSONL diffs cleanly line by line), spreadsheet
exports arrive as CSV, and YAML is what a human reaches for when annotating a
handful of examples by hand. All three normalize to the same
:class:`TrainingExample`, so nothing downstream knows which one was used.

Nothing here imports scikit-learn: a dataset is a domain object, and being able
to load and inspect one without the ML extra installed keeps the seed data
testable in the default environment.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from autopilot.domain.entities import TrainingExample
from autopilot.domain.enums import ExampleSource, TaskType, Tier
from autopilot.domain.errors import ConfigurationError
from autopilot.infrastructure.observability.logging import get_logger

_log = get_logger(__name__)

#: Extensions this module knows how to parse, mapped to a format label used in
#: error messages.
SUPPORTED_SUFFIXES: dict[str, str] = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".yaml": "yaml",
    ".yml": "yaml",
}

#: Column/key aliases accepted for the tier label, so a spreadsheet exported
#: with a different header does not require editing by hand.
_TIER_KEYS = ("tier", "label_tier", "label")
_PROMPT_KEYS = ("prompt", "text", "input")


@dataclass(frozen=True, slots=True)
class Dataset:
    """An ordered, deduplicated collection of labelled examples.

    Ordering is preserved from the source file and never shuffled in place, so
    :attr:`fingerprint` is reproducible for a given input.
    """

    examples: tuple[TrainingExample, ...]

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __iter__(self) -> Iterator[TrainingExample]:
        """Iterate the examples in source order."""
        return iter(self.examples)

    @property
    def fingerprint(self) -> str:
        """SHA-256 over the ordered ``(prompt_hash, tier)`` pairs.

        Recorded in classifier metadata so a trained artifact can always be
        traced back to the exact rows that produced it.
        """
        digest = sha256()
        for example in self.examples:
            digest.update(f"{example.prompt_hash}:{int(example.tier)}\n".encode())
        return digest.hexdigest()

    @property
    def tier_counts(self) -> dict[Tier, int]:
        """Number of examples per tier, including tiers with none."""
        counts = Counter(e.tier for e in self.examples)
        return {tier: counts.get(tier, 0) for tier in Tier}

    @property
    def tier_balance(self) -> dict[Tier, float]:
        """Fraction of the dataset held by each tier."""
        total = len(self.examples)
        if total == 0:
            return dict.fromkeys(Tier, 0.0)
        return {tier: count / total for tier, count in self.tier_counts.items()}

    def filter(self, *, holdout: bool | None = None) -> Dataset:
        """Return the subset matching ``holdout``, or the whole dataset when ``None``."""
        if holdout is None:
            return self
        return Dataset(tuple(e for e in self.examples if e.is_holdout is holdout))

    def prompts(self) -> tuple[str, ...]:
        """Return every prompt in order."""
        return tuple(e.prompt for e in self.examples)

    def labels(self) -> tuple[Tier, ...]:
        """Return every tier label in order."""
        return tuple(e.tier for e in self.examples)


def load_dataset(*paths: Path, dedupe: bool = True) -> Dataset:
    """Load and merge labelled examples from one or more files.

    Args:
        *paths: Files to read. Format is inferred from each suffix.
        dedupe: Drop later rows repeating an earlier ``(prompt_hash, tier)``.
            Earlier files win, so a curated seed set takes precedence over
            automatically appended production failures.

    Returns:
        The merged dataset, in the order the rows were read.

    Raises:
        ConfigurationError: A file is missing, of an unsupported format, or
            contains a row that cannot be interpreted as a labelled example.
    """
    merged: list[TrainingExample] = []
    for path in paths:
        merged.extend(_load_one(path))

    examples = _dedupe(merged) if dedupe else merged
    dropped = len(merged) - len(examples)
    dataset = Dataset(tuple(examples))
    _log.info(
        "dataset.loaded",
        files=[str(p) for p in paths],
        examples=len(dataset),
        duplicates_dropped=dropped,
        tier_counts={int(t): c for t, c in dataset.tier_counts.items()},
    )
    return dataset


def stratified_split(
    dataset: Dataset, *, holdout_fraction: float = 0.2, seed: int = 0
) -> tuple[Dataset, Dataset]:
    """Split ``dataset`` into train and holdout parts, preserving tier balance.

    The split is deterministic for a given ``seed``: rows are ordered within
    each tier by a hash of their content rather than by position, so adding an
    example to the middle of the seed file does not silently reshuffle which
    rows are held out.

    Args:
        dataset: The dataset to split.
        holdout_fraction: Share of each tier to reserve for evaluation.
        seed: Salt for the ordering hash.

    Returns:
        ``(train, holdout)``. Every returned holdout example is flagged
        ``is_holdout=True`` so the flag survives being written back to disk.

    Raises:
        ConfigurationError: ``holdout_fraction`` is not strictly between 0 and 1.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ConfigurationError(
            f"holdout_fraction must be between 0 and 1 exclusive, got {holdout_fraction}."
        )

    train: list[TrainingExample] = []
    holdout: list[TrainingExample] = []
    for tier in Tier:
        rows = [e for e in dataset if e.tier is tier]
        ordered = sorted(rows, key=lambda e: sha256(f"{seed}:{e.prompt_hash}".encode()).hexdigest())
        # round() rather than int(): with 5 rows at 20% this reserves 1, not 0.
        take = round(len(ordered) * holdout_fraction)
        holdout.extend(r.model_copy(update={"is_holdout": True}) for r in ordered[:take])
        train.extend(r.model_copy(update={"is_holdout": False}) for r in ordered[take:])

    _log.info("dataset.split", train=len(train), holdout=len(holdout), seed=seed)
    return Dataset(tuple(train)), Dataset(tuple(holdout))


def write_jsonl(dataset: Dataset, path: Path) -> None:
    """Write ``dataset`` to ``path`` as JSONL, one example per line.

    Args:
        dataset: The examples to write.
        path: Destination file; parent directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "prompt": e.prompt,
                "tier": int(e.tier),
                "task_type": e.task_type.value,
                "source": e.source.value,
                "is_holdout": e.is_holdout,
                **({"notes": e.notes} if e.notes else {}),
            },
            ensure_ascii=False,
        )
        for e in dataset
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("dataset.written", path=str(path), examples=len(dataset))


def _load_one(path: Path) -> list[TrainingExample]:
    """Parse a single file into examples, dispatching on its suffix."""
    fmt = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if fmt is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ConfigurationError(
            f"Unsupported dataset format {path.suffix!r} at {path}. Supported: {supported}."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Dataset not found at {path}") from exc

    rows = {
        "jsonl": _parse_jsonl,
        "json": _parse_json,
        "csv": _parse_csv,
        "yaml": _parse_yaml,
    }[fmt](text, path)

    return [_to_example(row, path=path, index=i) for i, row in enumerate(rows, start=1)]


def _parse_jsonl(text: str, path: Path) -> list[Any]:
    """Parse newline-delimited JSON, ignoring blank lines."""
    rows: list[Any] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"{path}:{number} is not valid JSON: {exc}") from exc
    return rows


def _parse_json(text: str, path: Path) -> list[Any]:
    """Parse a JSON document holding a list of rows, or ``{"examples": [...]}``."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{path} is not valid JSON: {exc}") from exc
    return _unwrap(document, path)


def _parse_yaml(text: str, path: Path) -> list[Any]:
    """Parse a YAML document holding a list of rows, or ``{"examples": [...]}``."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{path} is not valid YAML: {exc}") from exc
    return _unwrap(document, path)


def _parse_csv(text: str, path: Path) -> list[Any]:
    """Parse a CSV with a header row."""
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise ConfigurationError(f"{path} has no header row.")
    return list(reader)


def _unwrap(document: Any, path: Path) -> list[Any]:
    """Accept either a bare list or a mapping with an ``examples`` key."""
    if isinstance(document, dict):
        document = document.get("examples")
    if not isinstance(document, list):
        raise ConfigurationError(
            f"{path} must contain a list of examples, or a mapping with an 'examples' key."
        )
    return document


def _to_example(row: Any, *, path: Path, index: int) -> TrainingExample:
    """Validate one parsed row into a :class:`TrainingExample`.

    Raises:
        ConfigurationError: The row is not a mapping, is missing a prompt or
            tier, or carries a value outside the allowed vocabulary.
    """
    where = f"{path} row {index}"
    if not isinstance(row, dict):
        raise ConfigurationError(f"{where} must be a mapping, got {type(row).__name__}.")

    prompt = _first(row, _PROMPT_KEYS)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigurationError(f"{where} is missing a non-empty 'prompt'.")

    raw_tier = _first(row, _TIER_KEYS)
    if raw_tier is None or (isinstance(raw_tier, str) and not raw_tier.strip()):
        raise ConfigurationError(f"{where} is missing a 'tier' label.")
    try:
        tier = Tier(int(raw_tier))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{where} has an invalid tier {raw_tier!r}; expected one of {[int(t) for t in Tier]}."
        ) from exc

    try:
        task_type = TaskType(str(row.get("task_type") or TaskType.GENERAL.value).strip().lower())
        source = ExampleSource(str(row.get("source") or ExampleSource.SEED.value).strip().lower())
    except ValueError as exc:
        raise ConfigurationError(f"{where} has an unrecognized value: {exc}") from exc

    return TrainingExample(
        prompt=prompt.strip(),
        tier=tier,
        task_type=task_type,
        source=source,
        is_holdout=_as_bool(row.get("is_holdout")),
        notes=str(row.get("notes") or ""),
    )


def _first(row: dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first present, non-``None`` value among ``keys``."""
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def _as_bool(value: Any) -> bool:
    """Coerce a CSV/YAML truthy value to ``bool``.

    CSV has no types, so ``"false"`` arrives as a non-empty — and therefore
    truthy — string; treating it as ``True`` would leak holdout rows into
    training, which is the one mistake this dataset must never make.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _dedupe(examples: Iterable[TrainingExample]) -> list[TrainingExample]:
    """Drop repeats of an earlier ``(prompt_hash, tier)``, keeping the first."""
    seen: set[tuple[str, Tier]] = set()
    unique: list[TrainingExample] = []
    for example in examples:
        key = example.dedupe_key
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique
