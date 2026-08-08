
"""
Base classes for dataset loading and benchmark definitions.

CONCEPT: The EvalSample — Atomic Unit of Evaluation
──────────────────────────────────────────────────────
Every evaluation, regardless of task type, reduces to comparing a model's
output against a reference answer. The EvalSample holds one such pair.

Task types and how they map to EvalSample fields:

  ┌─────────────────────┬──────────────┬──────────────┬──────────┬───────┐
  │ Task                │ input        │ reference    │ choices  │ label │
  ├─────────────────────┼──────────────┼──────────────┼──────────┼───────┤
  │ Multiple-choice QA  │ question     │ answer text  │ [A,B,C,D]│ 0-3   │
  │ Open-ended QA       │ question     │ gold answer  │ None     │ None  │
  │ Summarization       │ document     │ summary      │ None     │ None  │
  │ Code generation     │ docstring    │ solution code│ None     │ None  │
  │ Classification      │ text         │ label name   │ None     │ 0-N   │
  └─────────────────────┴──────────────┴──────────────┴──────────┴───────┘

CONCEPT: Dataset = Collection of EvalSamples + Metadata
─────────────────────────────────────────────────────────
BaseDataset wraps a list of EvalSamples and provides:
  - Lazy loading (samples loaded on first access, not __init__)
  - Iteration, slicing, random sampling
  - DatasetInfo metadata (name, task_type, license, etc.)

All benchmarks (MMLU, GSM8K, HellaSwag) and custom datasets
extend BaseDataset and implement load() to return EvalSamples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class EvalSample:
    """
    A single evaluation example.

    Fields:
        id:        Unique ID within the dataset (e.g., "mmlu_anatomy_0042").
        input:     The prompt / question sent to the model.
        reference: Expected gold output for metric computation.
                   None for tasks where quality is judged by LLM-as-a-Judge only.
        choices:   Answer options for multiple-choice tasks, e.g. ["Paris", "London", ...].
        label:     Index into choices for the correct answer (0-based).
        context:   Additional document/passage (used in RAG eval, reading comprehension).
        metadata:  Arbitrary extras: difficulty, subject, source, split, etc.
    """
    id: str
    input: str
    reference: str | None = None
    choices: list[str] | None = None
    label: int | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self, template: str | None = None) -> str:
        """
        Format this sample into a ready-to-send prompt string.

        Args:
            template: Optional format string with placeholders:
                      {input}, {context}, {choices}

        Default (no template): returns self.input as-is.

        Example for multiple-choice:
            template = "{input}\\n\\nOptions:\\n{choices}\\n\\nAnswer:"
            sample.to_prompt(template)
            # "What is the capital of France?
            #
            #  Options:
            #  A. Berlin
            #  B. Paris
            #  C. Rome
            #  D. Madrid
            #
            #  Answer:"
        """
        if template is None:
            return self.input

        choices_text = ""
        
        if self.choices:
            choices_text = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(self.choices)
            )

        return template.format(
            input=self.input,
            context=self.context or "",
            choices=choices_text,
        )

@dataclass
class DatasetInfo:
    """Metadata about a dataset — shown in the UI and logged to MLflow."""
    name: str
    description: str
    task_type: str      # "multiple_choice" | "generation" | "qa" | "classification" | "code"
    size: int | None = None
    language: str = "en"
    license: str = "unknown"
    source: str = ""
    tags: list[str] = field(default_factory=list)

class BaseDataset(ABC):
    """
    Abstract base for all EvalForge datasets.

    Concrete subclasses implement load() — everything else is provided.

    Lazy loading: samples aren't fetched until first access. This means
    you can register a dataset object cheaply and only pay I/O cost
    when evaluation actually starts.

    Usage:
        dataset = MMLUDataset(subject="anatomy", split="test")
        print(len(dataset))          # triggers load() once
        sample = dataset[0]          # EvalSample
        for sample in dataset:       # iterate all
            ...
        subset = dataset.sample(100) # random 100 samples
    """

    def __init__(self, split: str = "test", seed: int = 42) -> None:
        self.split = split
        self.seed = seed
        self._samples: list[EvalSample] | None = None

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Dataset metadata. Implemented by each concrete dataset."""

    @abstractmethod
    def load(self) -> list[EvalSample]:
        """
        Load and return all EvalSamples for this dataset.

        Called once (lazily) on first access to `.samples`.
        Can download from HuggingFace Hub, read a local file, etc.
        """

    @property
    def samples(self) -> list[EvalSample]:
        """Lazily load and cache samples."""
        if self._samples is None:
            self._samples = self.load()
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int | slice) -> "EvalSample | list[EvalSample]":
        return self.samples[idx]

    def sample(self, n: int) -> list[EvalSample]:
        """Return n randomly sampled examples (seeded for reproducibility)."""
        import random
        rng = random.Random(self.seed)
        pool = self.samples
        return rng.sample(pool, min(n, len(pool)))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.info.name!r}, "
            f"split={self.split!r}, "
            f"size={len(self)})"
        )


class SimpleDataset(BaseDataset):
    """Wraps a pre-loaded list of EvalSamples — used by evaluation_service."""

    def __init__(self, name: str, samples: list[EvalSample], split: str = "custom") -> None:
        super().__init__(split=split)
        self._info = DatasetInfo(name=name, description="", task_type="generation")
        self._samples = samples

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def load(self) -> list[EvalSample]:
        return self._samples or []