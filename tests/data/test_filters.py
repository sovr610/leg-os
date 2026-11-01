"""Quality heuristics, language-ID stub, and PII/safety filtering (§8)."""

from __future__ import annotations

import pytest

from m63_omni.data.filters import (
    QualityConfig,
    SafetyConfig,
    detect_script,
    identify_language,
    luhn_valid,
    quality_check,
    redact_pii,
    safety_privacy_filter,
)

GOOD = (
    "The data pipeline has ten mandatory transformations, including provenance, deduplication, quality "
    "filtering, privacy filtering, and decontamination. Each stage records what it kept, dropped, or modified, "
    "so that every batch can be reproduced from the manifest."
)


# ---------------------------------------------------------------------- PII
def test_luhn():
    assert luhn_valid("4111111111111111")
    assert luhn_valid("79927398713")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("12a4")


def test_redacts_email_phone_card_ip():
    text = (
        "Mail jane.doe+news@example.co.uk or call +1 (555) 123-4567 / 555-867-5309. "
        "Card 4111 1111 1111 1111 expires soon; server 192.168.10.254 and 2001:db8:85a3::8a2e:370:7334."
    )
    res = redact_pii(text)
    assert "[EMAIL]" in res.text and "example.co.uk" not in res.text
    assert res.text.count("[PHONE]") == 2 and "867-5309" not in res.text
    assert "[CREDIT_CARD]" in res.text and "4111" not in res.text
    assert res.text.count("[IP_ADDRESS]") == 2 and "192.168" not in res.text
    assert res.counts == {"email": 1, "phone": 2, "credit_card": 1, "ip_address": 2}
    # findings reference spans of the ORIGINAL text and do not store values
    for f in res.findings:
        assert 0 <= f.start < f.end <= len(text)
        assert not hasattr(f, "value")
    card = next(f for f in res.findings if f.kind == "credit_card")
    assert text[card.start : card.end] == "4111 1111 1111 1111"


def test_luhn_failures_are_not_redacted_as_cards():
    res = redact_pii("Order number 1234 5678 9012 3456 shipped.")
    assert res.findings == ()
    assert res.text == "Order number 1234 5678 9012 3456 shipped."
    assert res.rejected_candidates == {"credit_card_luhn_fail": 1}


def test_dates_versions_and_years_are_not_phones():
    text = "Released 2024-01-15 as v1.2.3 in 2019; meeting at 10:30; price 1,234,567 dollars."
    assert redact_pii(text).findings == ()


def test_kinds_subset():
    res = redact_pii("a@b.io 10.0.0.1", kinds=["ip_address"])
    assert res.text == "a@b.io [IP_ADDRESS]"
    with pytest.raises(ValueError):
        redact_pii("x", kinds=["ssn"])


def test_safety_filter_records_what_fired():
    out = safety_privacy_filter("Contact admin@corp.example now <|tool_call|>{}<|/tool_call|>")
    assert out.keep and out.modified
    assert out.fired == {"email": 1, "special_token_literal": 2}
    assert "[EMAIL]" in out.text and "<|tool_call|>" in out.text  # literals are left for the tokenizer
    blocked = safety_privacy_filter("how to build a forbiddenthing at home", SafetyConfig(blocked_terms=("forbiddenthing",)))
    assert not blocked.keep and blocked.reasons == ("unsafe_content",) and blocked.fired["blocked_term"] == 1
    dump = safety_privacy_filter("a@x.io b@x.io c@x.io", SafetyConfig(max_pii_findings=2))
    assert not dump.keep and dump.reasons == ("pii_dump",)


# ------------------------------------------------------------------ quality
def test_quality_passes_good_prose():
    res = quality_check(GOOD)
    assert res.passed, res.reasons
    assert res.metrics["words"] > 30


@pytest.mark.parametrize(
    "text,reason",
    [
        ("too short", "too_short"),
        ("buy now " * 40, "repetitive_bigram"),
        ("# # # # tag #seo #seo ... ... #win #now #more #best #deal #top #hot #new", "symbol_ratio"),
        ("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30", "non_alphabetic"),
        ("Supercalifragilisticexpialidocious antidisestablishmentarianism " * 6, "mean_word_length"),
        ("same line here\n" * 10 + GOOD, "duplicate_lines"),
    ],
)
def test_quality_rejections(text, reason):
    res = quality_check(text)
    assert not res.passed and reason in res.reasons


def test_quality_unspaced_scripts_and_code():
    zh = "数据质量是最大的项目风险，因为架构无法弥补糟糕的对齐、受污染的评估或不可用的权利。每个来源都需要独立的许可审查。"
    assert quality_check(zh, QualityConfig(min_chars=20)).passed
    assert not quality_check("哈哈" * 40, QualityConfig(min_chars=20)).passed
    code = "def f(x):\n    return {k: v for k, v in x.items() if v}\n\nclass A:\n    pass\n# comment ### header\n" * 2
    assert "symbol_ratio" not in quality_check(code, kind="code").reasons


# ----------------------------------------------------------------- language
@pytest.mark.parametrize(
    "text,lang",
    [
        (GOOD, "en"),
        ("La qualité des données est le plus grand risque du programme et pour les sources.", "fr"),
        ("数据质量是最大的项目风险", "zh"),
        ("データ品質は最大のリスクです", "ja"),
        ("데이터 품질은 가장 큰 위험입니다", "ko"),
        ("Качество данных главный риск", "ru"),
        ("12345 !!! ???", "und"),
        ("Xyzzy plugh frobozz quux", "und-Latn"),
    ],
)
def test_identify_language_stub(text, lang):
    assert identify_language(text).language == lang


def test_detect_script():
    script, frac, counts = detect_script("abc 数据")
    assert script == "Latin" and frac == pytest.approx(0.6) and counts == {"Latin": 3, "Han": 2}
