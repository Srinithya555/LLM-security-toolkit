"""
Scans LLM-generated OUTPUT (not input) for sensitive data that shouldn't
be echoed back — OWASP LLM Top 10's "Insecure Output Handling" and
"Sensitive Information Disclosure" categories. This matters because LLMs
can inadvertently regurgitate training data, leak system-prompt content
they were told to keep confidential, or echo back sensitive data from
earlier in a conversation (e.g. a user pastes an API key for debugging
help, and a naive app logs/displays the full response including it).
"""
import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
AWS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
GENERIC_API_KEY_PATTERN = re.compile(r"\b(sk|pk|api)[-_][A-Za-z0-9]{20,}\b", re.IGNORECASE)
PRIVATE_KEY_HEADER_PATTERN = re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----")


@dataclass
class LeakageFinding:
    category: str
    description: str
    matched_text: str  # truncated/masked before display — see mask_finding()


def _luhn_checksum(card_number: str) -> bool:
    """
    Validates a candidate credit-card-shaped number against the Luhn
    algorithm, to cut down on false positives from other 13-16 digit
    numbers (order IDs, phone numbers, etc.) that aren't actually credit
    cards. Real credit card numbers satisfy this checksum; most
    coincidental digit strings won't.
    """
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def scan_output_for_leakage(text: str) -> list:
    findings = []

    for match in EMAIL_PATTERN.finditer(text):
        findings.append(LeakageFinding("email", "Email address present in output", match.group(0)))

    for match in CREDIT_CARD_PATTERN.finditer(text):
        candidate = match.group(0)
        if _luhn_checksum(candidate):
            findings.append(LeakageFinding("credit_card", "Luhn-valid credit card number present in output", candidate))

    for match in SSN_PATTERN.finditer(text):
        findings.append(LeakageFinding("ssn", "SSN-shaped number present in output", match.group(0)))

    for match in AWS_KEY_PATTERN.finditer(text):
        findings.append(LeakageFinding("aws_key", "AWS access key ID present in output", match.group(0)))

    for match in GENERIC_API_KEY_PATTERN.finditer(text):
        findings.append(LeakageFinding("api_key", "API-key-shaped string present in output", match.group(0)))

    if PRIVATE_KEY_HEADER_PATTERN.search(text):
        findings.append(LeakageFinding("private_key", "Private key PEM header present in output", "-----BEGIN ... PRIVATE KEY-----"))

    return findings


def mask_finding(finding: LeakageFinding) -> str:
    """Masks all but the first/last couple characters — for safely
    logging THAT a leak was caught without logging the leaked value itself."""
    text = finding.matched_text
    if len(text) <= 4:
        return "*" * len(text)
    return text[:2] + "*" * (len(text) - 4) + text[-2:]


def redact_output(text: str, findings: list) -> str:
    """Returns the output text with every flagged span replaced by a
    category-labeled placeholder — for actually withholding the sensitive
    content from the end user, not just logging that it was found."""
    redacted = text
    for finding in findings:
        redacted = redacted.replace(finding.matched_text, f"[REDACTED:{finding.category.upper()}]")
    return redacted
