import re
from collections import Counter
from evalforge.metrics.base import BaseMetric, MetricResult

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _squad_f1(pred: str, ref: str) -> float:
    """ SQuAD-style token F1 using token counts (not sets). """
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

_CHOICE_RE = re.compile(r"\b([A-Da-d])\b[\.\):\s]")

# Open-ended QA
class AnswerMatchMetric(BaseMetric):
    """
    Robust metric for open-ended QA.

    Tiers:
    - Exact match (after normalization)            → 1.0
    - Reference contained, concise answer          → 1.0
    - Reference contained, verbose/hedged answer   → 0.70–0.99  (verbosity penalty)
    - Reference not contained, partial token match → SQuAD F1  (0.0–0.69)
    - No token overlap                             → 0.0

    The verbosity penalty uses a 10× reference-length tolerance window.
    Answers within 10× the reference length are not penalised.
    Beyond that, the score decays toward the 0.70 floor.
    """

    @property
    def name(self) -> str:
        return "answer_match"

    def score(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        pred = _normalize(prediction)
        ref = _normalize(reference)

        if pred == ref:
            return MetricResult(name=self.name, score=1.0, details={"match_type": "exact"})

        if ref in pred:
            ref_len = max(len(ref.split()), 1)
            pred_len = max(len(pred.split()), 1)
            tolerance = ref_len * 10
            verbosity = min(1.0, tolerance / pred_len)
            score = round(0.70 + 0.30 * verbosity, 4)
            return MetricResult(name=self.name, score=score, details={"match_type": "contains", "verbosity_factor": round(verbosity, 4)})

        f1 = round(_squad_f1(pred, ref), 4)
        return MetricResult(name=self.name, score=f1, details={"match_type": "partial_f1"})

# Multiple-choice QA
class ChoiceMatchMetric(BaseMetric):
    """
    MCQ with exactly one correct answer.

    Resolution order:
    1. Extract A/B/C/D letter from prediction — compare to expected letter.
    2. Check if correct choice text is contained in prediction.
    3. Fall back to reference text exact/contains match.

    kwargs: choices (list[str]), label (int, 0-based index of correct choice)
    """

    @property
    def name(self) -> str:
        return "choice_match"

    def score(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        choices: list[str] = kwargs.get("choices") or []
        label: int | None = kwargs.get("label")
        pred_norm = _normalize(prediction)
        ref_norm = _normalize(reference)

        expected_letter: str | None = None
        
        if label is not None and 0 <= label < 26:
            expected_letter = chr(65 + label).lower()

        found_letters = [m.lower() for m in _CHOICE_RE.findall(prediction)]
        
        if found_letters and expected_letter:
            hit = found_letters[0] == expected_letter
            return MetricResult(name=self.name, score=1.0 if hit else 0.0, details={"strategy": "letter", "found": found_letters[0], "expected": expected_letter})

        if choices and label is not None and 0 <= label < len(choices):
            correct_text = _normalize(choices[label])
            
            if correct_text and correct_text in pred_norm:
                return MetricResult(name=self.name, score=1.0, details={"strategy": "choice_text"})

        if pred_norm == ref_norm or (ref_norm and ref_norm in pred_norm):
            return MetricResult(name=self.name, score=1.0, details={"strategy": "reference"})

        return MetricResult(name=self.name, score=0.0)


# MCQ - Multiple correct answers
class MultiChoiceMatchMetric(BaseMetric):
    """
    MCQ where multiple answers may be correct (select-all-that-apply).

    reference should encode expected correct letters or 0-based indices,
    comma/semicolon separated: e.g. "A,C" or "0,2".

    Score = Jaccard similarity between predicted letter set and expected set.
    """

    @property
    def name(self) -> str:
        return "multi_choice_match"

    def score(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        pred_letters = set(m.lower() for m in _CHOICE_RE.findall(prediction))
        expected: set[str] = set()
        
        for part in re.split(r"[,;\s]+", reference.strip()):
            part = part.strip().lower()
            
            if part in "abcdefghij":
                expected.add(part)
            elif part.isdigit():
                idx = int(part)
                
                if 0 <= idx < 26:
                    expected.add(chr(97 + idx))

        if not expected:
            return MetricResult(name=self.name, score=0.0, details={"error": "could not parse expected choices"})

        intersection = pred_letters & expected
        union = pred_letters | expected
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return MetricResult(name=self.name, score=round(jaccard, 4), details={"predicted": sorted(pred_letters), "expected": sorted(expected)})

# Generation/Summarization
class RougeLMetric(BaseMetric):
    """
    Rouge-L F1 via longest common subsequence.
    Best for generation tasks: summarization, translation, paraphrase.
    """

    @property
    def name(self) -> str:
        return "rouge_l"

    def score(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        lcs = self._lcs(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return MetricResult(name=self.name, score=round(f1, 4),
                            details={"precision": round(precision, 4),
                                     "recall": round(recall, 4)})

    @staticmethod
    def _lcs(a: list[str], b: list[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

# Code generation
_COMMENT_RE = re.compile(r"#.*$|//.*$|/\*[\s\S]*?\*/", re.MULTILINE)

class CodeMatchMetric(BaseMetric):
    """
    Token F1 for code generation.

    Strips comments, normalises whitespace, tokenises on word and operator
    boundaries, then computes SQuAD-style F1 on the resulting token stream.
    """

    @property
    def name(self) -> str:
        return "code_match"

    def score(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        pred = self._clean(prediction)
        ref = self._clean(reference)
        if pred == ref:
            return MetricResult(name=self.name, score=1.0)
        f1 = round(_squad_f1(pred, ref), 4)
        return MetricResult(name=self.name, score=f1)

    @staticmethod
    def _clean(code: str) -> str:
        code = _COMMENT_RE.sub("", code)
        code = re.sub(r"\s+", " ", code).strip().lower()
        return re.sub(r"([^\w\s])", r" \1 ", code)

# Classification
class LabelMatchMetric(BaseMetric):
    """
    Label accuracy for classification tasks.

    1. Exact match after normalization           → 1.0
    2. Label found as whole word in prediction   → 1.0  (handles "Class: positive")
    3. No match                                  → 0.0

    Whole-word boundary prevents "positive" matching inside "false positive".
    """

    @property
    def name(self) -> str:
        return "label_match"

    def score(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        pred = _normalize(prediction)
        ref = _normalize(reference)

        if pred == ref:
            return MetricResult(name=self.name, score=1.0, details={"match_type": "exact"})

        if re.search(rf"\b{re.escape(ref)}\b", pred):
            return MetricResult(name=self.name, score=1.0, details={"match_type": "word_boundary"})

        return MetricResult(name=self.name, score=0.0)

_NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")

class NumericMatchMetric(BaseMetric):
    """
    Exact numeric match for math QA (GSM8K, MATH, etc).

    Extracts the last number from prediction (handles chain-of-thought output),
    normalises commas and currency symbols, then compares as float.
    Binary: 1.0 if equal, 0.0 otherwise.
    """

    @property
    def name(self) -> str:
        return "numeric_match"

    @staticmethod
    def _extract_number(text: str) -> float | None:
        text = text.replace(",", "").replace("$", "").strip()
        matches = _NUMBER_RE.findall(text)
        if not matches:
            return None
        try:
            return float(matches[-1].replace(",", "").replace("$", ""))
        except ValueError:
            return None

    def score(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        pred_val = self._extract_number(prediction)
        ref_val = self._extract_number(reference)

        if pred_val is None or ref_val is None:
            return MetricResult(name=self.name, score=0.0, details={"error": "could not parse number"})

        match = abs(pred_val - ref_val) < 1e-6
        return MetricResult(
            name=self.name,
            score=1.0 if match else 0.0,
            details={"predicted": pred_val, "reference": ref_val},
        )

# Metric registry
METRIC_REGISTRY: dict[str, BaseMetric] = {
    "answer_match": AnswerMatchMetric(),       # open-ended QA
    "choice_match": ChoiceMatchMetric(),       # MCQ single answer
    "multi_choice_match": MultiChoiceMatchMetric(),  # MCQ multiple answers
    "rouge_l": RougeLMetric(),            # generation / summarization
    "code_match": CodeMatchMetric(),         # code generation
    "label_match": LabelMatchMetric(),        # classification
}

def get_metrics(names: list[str]) -> list[BaseMetric]:
    unknown = [name for name in names if name not in METRIC_REGISTRY]
    
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {list(METRIC_REGISTRY)}")
    
    return [METRIC_REGISTRY[name] for name in names]