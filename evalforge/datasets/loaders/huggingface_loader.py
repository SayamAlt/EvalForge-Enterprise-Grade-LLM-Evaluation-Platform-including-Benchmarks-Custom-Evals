"""
HuggingFace Hub dataset loader.

CONCEPT: HuggingFace datasets library
───────────────────────────────────────
`pip install datasets` gives you access to 100,000+ datasets hosted on
huggingface.co/datasets. You load them with:

    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "anatomy", split="test")
    # → Dataset of 135 rows, columns: question, choices, answer, subject

The library downloads and caches data in ~/.cache/huggingface/ on first call.
Subsequent calls are instant (from cache).

CONCEPT: Why converters per benchmark?
────────────────────────────────────────
Every benchmark was designed independently with different column schemas:

  MMLU:        question (str), choices (list[str]), answer (int 0-3), subject (str)
  GSM8K:       question (str), answer (str "Step 1... #### 42")
  HellaSwag:   activity_label, ctx, endings (list[str]), label (str "0"-"3")
  TruthfulQA:  question (str), mc2_targets {choices, labels}, best_answer

Each converter normalizes one benchmark schema → EvalSample.
Unknown datasets fall back to auto-detection (same logic as CSV loader).

CONCEPT: asyncio.to_thread
──────────────────────────
HF's load_dataset() is synchronous (it does blocking disk I/O and HTTP).
In an async FastAPI app, running synchronous I/O on the event loop blocks
ALL other requests until it completes — unacceptable.

Solution: asyncio.to_thread(fn, *args) runs fn in a thread pool, freeing
the event loop to handle other requests while waiting. The service layer
wraps this loader with to_thread().
"""

from typing import Any

from evalforge.datasets.base import EvalSample

# Maps HuggingFace dataset repo ID → converter name
# Add new entries here as we support more benchmarks
_KNOWN_SCHEMAS: dict[str, str] = {
    "cais/mmlu": "mmlu",
    "EleutherAI/hendrycks_test": "mmlu",
    "openai/gsm8k": "gsm8k",   # "gsm8k" bare ID deprecated; use openai/gsm8k
    "gsm8k": "gsm8k",          # kept for backwards compat with old cached refs
    "Rowan/hellaswag": "hellaswag",
    "truthful_qa": "truthful_qa",
    "openai/openai_humaneval": "humaneval",
    "cais/hle": "hle",
}


def _convert_mmlu(row: dict, idx: int, dataset_name: str) -> EvalSample:
    """
    MMLU: 57 academic subjects, 4-choice MC.
    answer is an int index into choices (0=A, 1=B, 2=C, 3=D).
    reference = the correct answer text (not just the index).
    """
    label = int(row["answer"])
    return EvalSample(
        id=f"{dataset_name}_{idx:06d}",
        input=row["question"],
        reference=row["choices"][label],
        choices=row["choices"],
        label=label,
        metadata={"subject": row.get("subject", "")},
    )


def _convert_gsm8k(row: dict, idx: int, dataset_name: str) -> EvalSample:
    """
    GSM8K: grade-school math word problems.
    answer format: "Step 1: ...\nStep 2: ...\n#### 42"
    We extract just the final number after "####" as the reference.
    The full chain-of-thought goes into metadata for reference.
    """
    full_answer = row["answer"]
    parts = full_answer.split("####")
    numeric_answer = parts[-1].strip() if len(parts) > 1 else full_answer.strip()
    return EvalSample(
        id=f"{dataset_name}_{idx:06d}",
        input=row["question"],
        reference=numeric_answer,
        metadata={"chain_of_thought": full_answer},
    )


def _convert_hellaswag(row: dict, idx: int, dataset_name: str) -> EvalSample:
    """
    HellaSwag: commonsense NLI — pick the most plausible sentence completion.
    label is a string "0"/"1"/"2"/"3" (int-as-string).
    """
    label_str = row.get("label", "")
    label = int(label_str) if label_str.isdigit() else None
    context = f"{row.get('activity_label', '')}: {row.get('ctx', '')}".strip(": ")
    endings: list[str] = row.get("endings", [])
    return EvalSample(
        id=f"{dataset_name}_{idx:06d}",
        input=context,
        reference=endings[label] if label is not None and label < len(endings) else None,
        choices=endings,
        label=label,
    )


def _convert_truthful_qa(row: dict, idx: int, dataset_name: str) -> EvalSample:
    """
    TruthfulQA MC2 mode: multiple correct answers, each scored 0 or 1.
    mc2_targets = {"choices": [...], "labels": [0, 1, 0, 1, ...]}
    """
    mc2 = row.get("mc2_targets", {})
    choices: list[str] = mc2.get("choices", [])
    labels: list[int] = mc2.get("labels", [])
    return EvalSample(
        id=f"{dataset_name}_{idx:06d}",
        input=row["question"],
        reference=row.get("best_answer", ""),
        choices=choices,
        metadata={"mc2_labels": labels, "category": row.get("category", "")},
    )


def _convert_humaneval(row: dict, idx: int, dataset_name: str) -> EvalSample:
    """
    HumanEval: Python coding problems.
    input = the function docstring + signature (the prompt shown to the model).
    reference = canonical_solution (the ground truth code to compare against).
    """
    return EvalSample(
        id=row.get("task_id", f"{dataset_name}_{idx:06d}"),
        input=row["prompt"],
        reference=row.get("canonical_solution"),
        metadata={
            "entry_point": row.get("entry_point"),
            "test": row.get("test"),
        },
    )


def _convert_hle(row: dict, idx: int, dataset_name: str) -> EvalSample | None:
    """
    HLE (Humanity's Last Exam): 2500 expert-level questions across all domains.
    Gated dataset — requires HF token + accepted terms at huggingface.co/cais/hle.

    Actual schema (verified): id, question, image (str — base64 data URI or ""),
    image_preview (PIL|None), answer, answer_type, author_name, rationale,
    rationale_image (PIL|None), raw_subject, category, canary

    Text-only rows have image == "" (empty string).
    Image rows have image == "data:image/...;base64,..." — skip those.
    """
    if row.get("image"):  # non-empty string means image present — skip
        return None

    question = row.get("question", "")
    if not question.strip():
        return None

    return EvalSample(
        id=row.get("id") or f"{dataset_name}_{idx:06d}",
        input=question,
        reference=str(row.get("answer", "")).strip(),
        metadata={
            "category": row.get("category", ""),
            "author_name": row.get("author_name", ""),
            "answer_type": row.get("answer_type", ""),
            "raw_subject": row.get("raw_subject", ""),
        },
    )


_CONVERTERS = {
    "mmlu": _convert_mmlu,
    "gsm8k": _convert_gsm8k,
    "hellaswag": _convert_hellaswag,
    "truthful_qa": _convert_truthful_qa,
    "humaneval": _convert_humaneval,
    "hle": _convert_hle,
}


def _auto_convert(
    row: dict,
    idx: int,
    dataset_name: str,
    format_config: dict | None,
) -> EvalSample:
    """
    Fallback for unknown HF datasets: re-use CSV loader's auto-detection
    since both deal with dict rows and column aliases.
    """
    from evalforge.datasets.loaders.csv_loader import _detect_column_mapping, _row_to_sample

    mapping = _detect_column_mapping(list(row.keys()))
    if format_config:
        mapping = {**mapping, **format_config}
    return _row_to_sample(row, mapping, idx, dataset_name)


def load_from_huggingface(
    hf_dataset_id: str,
    dataset_name: str,
    hf_config: str | None = None,
    hf_split: str = "test",
    n_samples: int | None = None,
    format_config: dict | None = None,
) -> list[EvalSample]:
    """
    Load a HuggingFace Hub dataset and return EvalSamples.

    This is SYNCHRONOUS — the caller must run it in a thread pool:
        samples = await asyncio.to_thread(load_from_huggingface, ...)

    Args:
        hf_dataset_id: HF repo ID, e.g. "cais/mmlu"
        dataset_name:  Used to prefix sample IDs
        hf_config:     Subset/config, e.g. "anatomy" for MMLU
        hf_split:      "test", "train", or "validation"
        n_samples:     Limit to first N samples (None = full dataset)
        format_config: Column overrides for auto-detection fallback

    Returns:
        List of EvalSample
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` package is required. Run: uv pip install datasets"
        ) from e

    from app.core.config import settings
    hf_token = settings.huggingface_api_token or None

    hf_ds = load_dataset(
        hf_dataset_id,
        hf_config,
        split=hf_split,
        trust_remote_code=False,
        token=hf_token,
    )

    if n_samples is not None:
        hf_ds = hf_ds.select(range(min(n_samples, len(hf_ds))))

    schema_key = _KNOWN_SCHEMAS.get(hf_dataset_id)
    converter = _CONVERTERS.get(schema_key) if schema_key else None

    samples: list[EvalSample] = []
    for idx, row in enumerate(hf_ds):
        row_dict: dict[str, Any] = dict(row)
        if converter is not None:
            sample = converter(row_dict, idx, dataset_name)
        else:
            sample = _auto_convert(row_dict, idx, dataset_name, format_config)
        if sample is not None:  # converters may return None to skip rows (e.g. image questions)
            samples.append(sample)

    return samples
