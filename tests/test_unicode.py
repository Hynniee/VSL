"""
Tests for unicode_utils.compose_char() and tokens_to_text().
Run: pytest tests/test_unicode.py
"""
import pytest

from src.unicode_utils import compose_char, tokens_to_text


@pytest.mark.parametrize(
    "base, modifier, tone, expected",
    [
        ("chu_a", "dau_mu", "thanh_sac", "ấ"),
        ("chu_a", "dau_trang", None, "ă"),
        ("chu_o", "dau_rau", "thanh_huyen", "ờ"),
        ("đ", None, None, "đ"),
        ("chu_u", "dau_rau", "thanh_nang", "ự"),
    ],
)
def test_compose_char(base, modifier, tone, expected):
    assert compose_char(base, modifier, tone) == expected


def test_tokens_to_text_simple():
    tokens = ["chu_x", "chu_i", "space", "chu_n"]
    assert tokens_to_text(tokens) == "xi n"


def test_tokens_to_text_empty():
    assert tokens_to_text([]) == ""


def test_tokens_to_text_only_spaces():
    assert tokens_to_text(["space", "khoang_trang", " "]).strip() == ""
