"""
CSV and JSONL loader — normalizes flat files into EvalSamples.

CONCEPT: Column mapping
────────────────────────
Every benchmark and custom dataset uses different column names:
  MMLU CSV:    question, choices, answer
  Custom QA:   prompt, expected_output
  Label tool:  input_text, gold_label

All of these must become EvalSample(input=..., reference=...).

We solve this with two-pass mapping:
  1. Auto-detect: scan column names against known aliases
  2. Manual override: user provides {"input_column": "my_col_name"}

CONCEPT: choices column formats
──────────────────────────────────
Multiple-choice answers can be stored differently:
  JSON array:  '["Paris","London","Berlin","Madrid"]'
  Pipe-delimited: "Paris|London|Berlin|Madrid"

We handle both automatically.
"""

import csv
import io
import json
from typing import Any

from evalforge.datasets.base import EvalSample

# Aliases: EvalSample field → list of common CSV column names
COLUMN_ALIASES: dict[str, list[str]] = {
    "input": ["input", "question", "prompt", "text", "instruction", "query", "sentence"],
    "reference": ["reference", "answer", "output", "response", "target", "label", "expected", "gold"],
    "choices": ["choices", "options", "candidates", "answers"],
    "label": ["label", "answer_index", "correct", "correct_index", "target_index", "answer_key"],
    "context": ["context", "passage", "document", "background", "article"],
}


def _detect_column_mapping(headers: list[str]) -> dict[str, str | None]:
    """
    Auto-detect column mappings from CSV headers.

    Tries each alias in order, case-insensitive. First match wins.
    Returns e.g. {"input_column": "question", "reference_column": "answer", ...}
    None values mean no matching column was found.
    """
    headers_lower = {h.strip().lower(): h for h in headers}
    mapping: dict[str, str | None] = {}
    for field, aliases in COLUMN_ALIASES.items():
        matched = next((headers_lower[a] for a in aliases if a in headers_lower), None)
        mapping[f"{field}_column"] = matched
    return mapping


def _resolve_mapping(headers: list[str], format_config: dict | None) -> dict[str, str | None]:
    """Merge auto-detected mapping with user overrides. User values take priority."""
    auto = _detect_column_mapping(headers)
    if not format_config:
        return auto
    return {**auto, **format_config}


def _parse_choices(raw: Any) -> list[str] | None:
    """Parse choices from a cell — handles JSON arrays and pipe-separated strings."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(c) for c in raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                return [str(c) for c in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        if "|" in raw:
            return [c.strip() for c in raw.split("|")]
        return None
    return None


def _row_to_sample(
    row: dict[str, Any],
    mapping: dict[str, str | None],
    idx: int,
    dataset_name: str,
) -> EvalSample:
    """Convert one CSV/JSONL row → EvalSample using the resolved column mapping."""

    def get(field: str) -> Any:
        col = mapping.get(f"{field}_column")
        return row.get(col) if col else None

    raw_label = get("label")
    label: int | None = None
    if raw_label is not None:
        try:
            label = int(raw_label)
        except (ValueError, TypeError):
            pass

    # Collect everything that wasn't mapped into metadata
    known_cols = {v for v in mapping.values() if v}
    metadata = {k: v for k, v in row.items() if k not in known_cols}

    return EvalSample(
        id=f"{dataset_name}_{idx:06d}",
        input=str(get("input") or ""),
        reference=str(get("reference")) if get("reference") is not None else None,
        choices=_parse_choices(get("choices")),
        label=label,
        context=str(get("context")) if get("context") is not None else None,
        metadata=metadata,
    )


def load_csv(
    content: bytes,
    dataset_name: str,
    format_config: dict | None = None,
) -> list[EvalSample]:
    """
    Parse CSV file bytes → list of EvalSamples.

    Args:
        content:      Raw file bytes (from UploadFile.read())
        dataset_name: Used to prefix sample IDs, e.g. "my_dataset_000001"
        format_config: Optional column overrides, e.g. {"input_column": "q"}

    Raises:
        ValueError if the file has no headers or no recognizable input column.
    """
    text = content.decode("utf-8-sig")  # utf-8-sig strips BOM if Excel-generated
    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    headers = [h.strip() for h in (reader.fieldnames or [])]
    reader.fieldnames = headers

    if not headers:
        raise ValueError("CSV file has no header row. First row must contain column names.")

    mapping = _resolve_mapping(headers, format_config)

    if not mapping.get("input_column"):
        raise ValueError(
            f"Cannot find an input column in CSV headers {headers}. "
            f"Expected one of: {COLUMN_ALIASES['input']}. "
            f"Use the input_column form field to specify the correct name."
        )

    return [
        _row_to_sample(dict(row), mapping, idx, dataset_name)
        for idx, row in enumerate(reader)
    ]


def load_jsonl(
    content: bytes,
    dataset_name: str,
    format_config: dict | None = None,
) -> list[EvalSample]:
    """
    Parse JSONL file bytes → list of EvalSamples.

    JSONL = one JSON object per line (not a JSON array).
    Empty lines are skipped.

    Raises:
        ValueError if the file is empty or contains invalid JSON.
    """
    lines = content.decode("utf-8").strip().split("\n")
    rows: list[dict] = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i}: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"Line {i} is not a JSON object (got {type(obj).__name__})")
        rows.append(obj)

    if not rows:
        raise ValueError("JSONL file is empty.")

    headers = [k.strip() for k in rows[0].keys()]
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    mapping = _resolve_mapping(headers, format_config)

    if not mapping.get("input_column"):
        raise ValueError(
            f"Cannot find an input field in JSONL. "
            f"Fields found: {headers}. "
            f"Expected one of: {COLUMN_ALIASES['input']}."
        )

    return [
        _row_to_sample(row, mapping, idx, dataset_name)
        for idx, row in enumerate(rows)
    ]
