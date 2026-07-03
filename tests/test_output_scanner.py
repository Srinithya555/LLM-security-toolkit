import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_security.output_scanner import scan_output_for_leakage, mask_finding, redact_output, _luhn_checksum


class TestEmailDetection:
    def test_flags_email(self):
        findings = scan_output_for_leakage("Contact me at alice@example.com")
        assert any(f.category == "email" for f in findings)

    def test_clean_text_not_flagged(self):
        assert scan_output_for_leakage("This has no sensitive data at all") == []


class TestLuhnChecksum:
    def test_known_valid_test_card_passes(self):
        # 4111111111111111 is a well-known PUBLIC test Visa number, not a real card
        assert _luhn_checksum("4111111111111111") is True

    def test_random_non_card_number_fails(self):
        assert _luhn_checksum("1234567890123456") is False

    def test_too_short_fails(self):
        assert _luhn_checksum("12345") is False


class TestCreditCardDetection:
    def test_luhn_valid_card_flagged(self):
        findings = scan_output_for_leakage("Card: 4111111111111111")
        assert any(f.category == "credit_card" for f in findings)

    def test_luhn_invalid_number_not_flagged_as_card(self):
        """Reduces false positives from order IDs, phone numbers, etc."""
        findings = scan_output_for_leakage("Order ID: 1234567890123456")
        assert not any(f.category == "credit_card" for f in findings)


class TestSecretDetection:
    def test_flags_aws_key(self):
        findings = scan_output_for_leakage("Key: AKIAIOSFODNN7EXAMPLE")
        assert any(f.category == "aws_key" for f in findings)

    def test_flags_generic_api_key(self):
        findings = scan_output_for_leakage("Use this key: sk-abc123def456ghi789jkl012")
        assert any(f.category == "api_key" for f in findings)

    def test_flags_private_key_header(self):
        findings = scan_output_for_leakage("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...")
        assert any(f.category == "private_key" for f in findings)

    def test_flags_ssn(self):
        findings = scan_output_for_leakage("SSN: 123-45-6789")
        assert any(f.category == "ssn" for f in findings)


class TestMaskingAndRedaction:
    def test_mask_finding_hides_middle_characters(self):
        from llm_security.output_scanner import LeakageFinding
        finding = LeakageFinding("email", "test", "alice@example.com")
        masked = mask_finding(finding)
        assert masked.startswith("al")
        assert masked.endswith("om")
        assert "alice@example" not in masked

    def test_redact_output_removes_sensitive_spans(self):
        text = "Contact alice@example.com for help"
        findings = scan_output_for_leakage(text)
        redacted = redact_output(text, findings)
        assert "alice@example.com" not in redacted
        assert "[REDACTED:EMAIL]" in redacted

    def test_redact_preserves_non_sensitive_text(self):
        text = "Contact alice@example.com for help"
        findings = scan_output_for_leakage(text)
        redacted = redact_output(text, findings)
        assert "for help" in redacted
