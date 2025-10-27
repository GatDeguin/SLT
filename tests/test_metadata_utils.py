"""Tests for metadata utility helpers."""

import pytest

from slt.utils.metadata import parse_split_column


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_parse_split_column_returns_empty_for_blank_values(raw):
    assert parse_split_column(raw) == []


def test_parse_split_column_parses_segments_and_normalises_quotes():
    raw = "[(\"intro “especial”\", 1.5, 3.25), ('cierre', '4,0', '5,5')]"

    segments = parse_split_column(raw)

    assert len(segments) == 2
    assert segments[0].text == 'intro "especial"'
    assert segments[0].start == pytest.approx(1.5)
    assert segments[0].end == pytest.approx(3.25)
    assert segments[1].text == "cierre"
    assert segments[1].start == pytest.approx(4.0)
    assert segments[1].end == pytest.approx(5.5)


def test_parse_split_column_rejects_invalid_payload():
    with pytest.raises(ValueError):
        parse_split_column("{'invalid': 'structure'}")
