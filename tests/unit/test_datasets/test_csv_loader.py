"""
Unit tests for CSV and JSONL loaders.

CONCEPT: Unit tests vs integration tests
─────────────────────────────────────────
Unit tests: test a single function with no external dependencies.
  - No DB, no HTTP, no HuggingFace network calls
  - Fast (< 1ms per test)
  - Tell you exactly which logic path broke

Integration tests: test a full request through the system.
  - Needs a real DB (or test DB)
  - Slower (100-500ms per test)
  - Tell you the system wires together correctly

For loaders: pure functions (bytes → EvalSamples), perfect for unit tests.
"""

import json
import pytest
from evalforge.datasets.loaders.csv_loader import load_csv, load_jsonl


# ── CSV tests ──────────────────────────────────────────────────────────────────

def _make_csv(*rows: str) -> bytes:
    return "\n".join(rows).encode()


def test_csv_basic_load():
    """Standard CSV with recognizable column names auto-detected."""
    csv = _make_csv(
        "question,answer",
        "What is 2+2?,4",
        "Capital of France?,Paris",
    )
    samples = load_csv(csv, "test_ds")
    assert len(samples) == 2
    assert samples[0].input == "What is 2+2?"
    assert samples[0].reference == "4"
    assert samples[1].input == "Capital of France?"
    assert samples[0].id == "test_ds_000000"


def test_csv_multiple_choice():
    """Multiple-choice CSV with choices as JSON array string."""
    csv = _make_csv(
        'question,choices,correct_index',
        '"What is 2+2?","[\'2\',\'4\',\'6\',\'8\']",1',
    )
    samples = load_csv(csv, "mc_ds")
    assert samples[0].choices == ["2", "4", "6", "8"]
    assert samples[0].label == 1


def test_csv_pipe_delimited_choices():
    """Choices stored as pipe-separated string."""
    csv = _make_csv(
        "question,answer,options,correct",
        "Pick one,B,A|B|C|D,1",
    )
    samples = load_csv(csv, "ds", format_config={"choices_column": "options", "label_column": "correct"})
    assert samples[0].choices == ["A", "B", "C", "D"]
    assert samples[0].label == 1


def test_csv_manual_column_mapping():
    """Non-standard column names resolved via format_config override."""
    csv = _make_csv(
        "my_q,gold_ans",
        "Hello?,World",
    )
    samples = load_csv(csv, "ds", format_config={"input_column": "my_q", "reference_column": "gold_ans"})
    assert samples[0].input == "Hello?"
    assert samples[0].reference == "World"


def test_csv_unknown_columns_go_to_metadata():
    """Columns not mapped to EvalSample fields are preserved in metadata."""
    csv = _make_csv(
        "question,answer,difficulty,source",
        "Q1?,A1,hard,wikipedia",
    )
    samples = load_csv(csv, "ds")
    assert samples[0].metadata["difficulty"] == "hard"
    assert samples[0].metadata["source"] == "wikipedia"


def test_csv_no_headers_raises():
    with pytest.raises(ValueError, match="no header row"):
        load_csv(b"", "ds")


def test_csv_no_input_column_raises():
    """Raises if no recognizable input column and no override provided."""
    csv = _make_csv("col_a,col_b", "x,y")
    with pytest.raises(ValueError, match="input column"):
        load_csv(csv, "ds")


# ── JSONL tests ────────────────────────────────────────────────────────────────

def test_jsonl_basic_load():
    lines = [
        json.dumps({"question": "Q1", "answer": "A1"}),
        json.dumps({"question": "Q2", "answer": "A2"}),
    ]
    content = "\n".join(lines).encode()
    samples = load_jsonl(content, "jds")
    assert len(samples) == 2
    assert samples[0].input == "Q1"
    assert samples[1].reference == "A2"


def test_jsonl_skips_blank_lines():
    content = b'{"question": "Q1", "answer": "A1"}\n\n{"question": "Q2", "answer": "A2"}\n'
    samples = load_jsonl(content, "ds")
    assert len(samples) == 2


def test_jsonl_invalid_json_raises():
    content = b'{"question": "Q1"}\nnot_json\n'
    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        load_jsonl(content, "ds")


def test_jsonl_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        load_jsonl(b"", "ds")
