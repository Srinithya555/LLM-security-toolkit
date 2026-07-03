"""
Pattern-based detection for prompt injection and jailbreak attempts
against LLM applications — the fast, interpretable, "known patterns"
layer, mapped to OWASP's Top 10 for LLM Applications (LLM01: Prompt
Injection). This mirrors the same layered-defense philosophy used
elsewhere in this portfolio (the RBAC+ABAC project's "coarse layer +
fine layer" split): heuristics catch known patterns cheaply and
explainably; ml_classifier.py in this same package catches variations
heuristics miss, at the cost of needing training data and occasionally
being wrong in ways that are harder to explain. Neither layer alone is
sufficient — see engine.py for how they're combined.

NOTE: OWASP's LLM Top 10 has had multiple revisions since first
published; verify current category numbering against
https://owasp.org/www-project-top-10-for-large-language-model-applications/
before citing it formally, same caution given for the CIS/MITRE
references elsewhere in this portfolio.
"""
import base64
import re
from dataclasses import dataclass


@dataclass
class InjectionSignal:
    rule_id: str
    category: str
    description: str
    matched_text: str = ""


_INSTRUCTION_OVERRIDE_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior|above|earlier) instructions", re.IGNORECASE),
    re.compile(r"disregard (all |the )?(previous|prior|above) (instructions|rules|prompt)", re.IGNORECASE),
    re.compile(r"forget (everything|all|what) (you were|i) (told|said)", re.IGNORECASE),
    re.compile(r"new instructions?:", re.IGNORECASE),
    re.compile(r"override your (instructions|programming|guidelines)", re.IGNORECASE),
]

_JAILBREAK_PERSONA_PATTERNS = [
    re.compile(r"\byou are now\b.{0,30}\b(DAN|unrestricted|jailbroken|uncensored)\b", re.IGNORECASE),
    re.compile(r"act as (an? )?(unrestricted|uncensored|amoral|jailbroken)", re.IGNORECASE),
    re.compile(r"pretend (you have|to have) no (restrictions|rules|guidelines|filters)", re.IGNORECASE),
    re.compile(r"do anything now", re.IGNORECASE),
    re.compile(r"developer mode", re.IGNORECASE),
]

_SYSTEM_PROMPT_EXTRACTION_PATTERNS = [
    re.compile(r"(repeat|show|print|reveal|what (is|are)|tell me) your (system prompt|instructions|initial prompt)", re.IGNORECASE),
    re.compile(r"what (were you|was your) (told|instructed|programmed)", re.IGNORECASE),
    re.compile(r"output (the text|everything) (above|before) this", re.IGNORECASE),
]

_FAKE_DELIMITER_PATTERNS = [
    # Attempts to inject fake conversation-turn markers to confuse the
    # model about where the "real" system/assistant turn begins.
    re.compile(r"\[?/?(system|assistant)\]?\s*:", re.IGNORECASE),
    re.compile(r"###\s*(system|instruction)", re.IGNORECASE),
    re.compile(r"<\|?(system|im_start|im_end)\|?>", re.IGNORECASE),
]


def _check_patterns(text: str, patterns: list, rule_id: str, category: str, description: str) -> list:
    signals = []
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            signals.append(InjectionSignal(rule_id=rule_id, category=category, description=description, matched_text=match.group(0)))
            break  # one signal per category is enough; avoid redundant near-duplicate signals
    return signals


def _check_suspicious_base64(text: str) -> list:
    """
    Flags long base64-looking substrings in user input — a real technique
    for smuggling instructions past naive keyword filters (encode the
    injection payload, ask the model to base64-decode and follow it).
    Heuristic, not proof: legitimate use (e.g. a user pasting a base64
    blob to ask "what is this") will also match — see engine.py for how
    this is weighted rather than treated as a hard block on its own.
    """
    candidates = re.findall(r"[A-Za-z0-9+/]{40,}={0,2}", text)
    signals = []
    for candidate in candidates:
        try:
            decoded = base64.b64decode(candidate, validate=True)
            decoded_text = decoded.decode("utf-8", errors="ignore")
            if any(kw in decoded_text.lower() for kw in ("ignore", "system", "instruction", "jailbreak")):
                signals.append(InjectionSignal(
                    rule_id="INJ-005", category="encoded_payload",
                    description="Base64-encoded text decodes to contain instruction-override keywords",
                    matched_text=candidate[:40] + "...",
                ))
        except Exception:
            continue
    return signals


def detect_injection_signals(text: str) -> list:
    signals = []
    signals += _check_patterns(text, _INSTRUCTION_OVERRIDE_PATTERNS, "INJ-001", "instruction_override",
                                 "Attempts to override or cancel prior instructions")
    signals += _check_patterns(text, _JAILBREAK_PERSONA_PATTERNS, "INJ-002", "jailbreak_persona",
                                 "Attempts to adopt an unrestricted/jailbroken persona")
    signals += _check_patterns(text, _SYSTEM_PROMPT_EXTRACTION_PATTERNS, "INJ-003", "prompt_extraction",
                                 "Attempts to extract the system prompt or hidden instructions")
    signals += _check_patterns(text, _FAKE_DELIMITER_PATTERNS, "INJ-004", "fake_delimiter",
                                 "Injects fake conversation-turn delimiters to confuse role boundaries")
    signals += _check_suspicious_base64(text)
    return signals
