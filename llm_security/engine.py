"""
Combines the heuristic pattern detector and the trained ML classifier
into one input-side risk assessment, plus wraps the output-side leakage
scanner — mirroring the layered-defense pattern used elsewhere in this
portfolio (RBAC+ABAC: coarse rule + fine-grained check; this project:
fast/interpretable heuristics + ML that catches rewordings heuristics miss).

Neither layer alone is sufficient: heuristics miss novel phrasings; the
ML classifier (trained on ~60 examples) will make mistakes on inputs
very different from its training distribution and offers no signal on
WHY it decided something, which matters for a human reviewing a flagged
prompt. Combining both gives defense in depth with explainability
preserved for the cases heuristics catch.
"""
from dataclasses import dataclass, field
from llm_security.prompt_injection_detector import detect_injection_signals
from llm_security.output_scanner import scan_output_for_leakage


@dataclass
class PromptAssessment:
    text: str
    heuristic_signals: list = field(default_factory=list)
    ml_label: str = None
    ml_confidence: float = None
    risk_level: str = "low"


def assess_prompt(text: str, vectorizer=None, model=None) -> PromptAssessment:
    heuristic_signals = detect_injection_signals(text)

    ml_label, ml_confidence = None, None
    if vectorizer is not None and model is not None:
        from llm_security.ml_classifier import predict
        result = predict(text, vectorizer, model)
        ml_label, ml_confidence = result["label"], result["confidence"]

    risk_level = _compute_risk_level(heuristic_signals, ml_label, ml_confidence)

    return PromptAssessment(
        text=text, heuristic_signals=heuristic_signals,
        ml_label=ml_label, ml_confidence=ml_confidence, risk_level=risk_level,
    )


def _compute_risk_level(heuristic_signals: list, ml_label: str, ml_confidence: float) -> str:
    """
    Simple combination rule, deliberately conservative (biased toward
    flagging over missing, appropriate for a security control): ANY
    heuristic match is at least "medium," a heuristic match PLUS ML
    agreement is "high," and ML flagging alone (no heuristic match) is
    "low-medium."

    BUG FIX (found by running the demo script, not by inspection): the
    first version of this function required ml_confidence > 0.6 before
    even reaching "low-medium," which meant a prompt the classifier
    correctly labeled "injection" at confidence 0.56 was silently
    downgraded all the way to "low" risk — a true positive getting
    under-reported because of an arbitrary threshold. The model already
    crossed its decision boundary to predict "injection" at all; that
    decision itself is the signal worth surfacing, not a threshold on top
    of it. Confidence is still surfaced to the caller in
    PromptAssessment.ml_confidence for anyone who wants finer-grained
    triage, but it no longer gates whether an ML injection flag is
    reported at all.
    """
    has_heuristic = len(heuristic_signals) > 0
    ml_flags_injection = ml_label == "injection"

    if has_heuristic and ml_flags_injection:
        return "high"
    if has_heuristic:
        return "medium"
    if ml_flags_injection:
        return "low-medium"
    return "low"


def assess_output(text: str) -> dict:
    findings = scan_output_for_leakage(text)
    return {
        "findings": findings,
        "should_block": len(findings) > 0,
        "categories_found": sorted({f.category for f in findings}),
    }
