"""Tests for the Streamlit GUI ``render_field`` default-value resolution.

These exercise the fallback chain (``examples`` → ``default`` → ``minimum`` →
fallback) that prevents ``TypeError: float() argument must be a string or a
number, not 'NoneType'`` when a schema field omits ``examples``/``default``/
``minimum`` (e.g. the ``ffn_activation_config`` learnable-activation params).
"""

from __future__ import annotations

import pytest

from src.streamlit_gui import app


@pytest.fixture(autouse=True)
def captured(monkeypatch):
    """Replace Streamlit widgets with recorders so no live server is needed."""
    captured = {}

    def number_input(label, value=None, min_value=None, max_value=None, **kwargs):
        captured["number_input"] = {
            "value": value,
            "min_value": min_value,
            "max_value": max_value,
        }
        return value

    def text_input(label, value="", **kwargs):
        captured["text_input"] = {"value": value}
        return value

    def warning(message):
        captured["warning"] = message

    monkeypatch.setattr(app.st, "number_input", number_input)
    monkeypatch.setattr(app.st, "text_input", text_input)
    monkeypatch.setattr(app.st, "warning", warning)
    return captured


def _number_schema(**overrides):
    schema = {"type": "number"}
    schema.update(overrides)
    return schema


def _integer_schema(**overrides):
    schema = {"type": "integer"}
    schema.update(overrides)
    return schema


def test_number_uses_examples_when_present(captured):
    app.render_field("x", _number_schema(examples=[0.5], default=0.25))
    assert captured["number_input"]["value"] == 0.5


def test_number_uses_default_when_no_examples(captured):
    app.render_field("x", _number_schema(default=0.25))
    assert captured["number_input"]["value"] == 0.25
    assert "warning" not in captured


def test_number_uses_minimum_when_no_examples_or_default(captured):
    app.render_field("x", _number_schema(minimum=0.1))
    assert captured["number_input"]["value"] == 0.1
    assert "warning" not in captured


def test_number_falls_back_and_warns_when_metadata_missing(captured):
    app.render_field("x", _number_schema())
    assert captured["number_input"]["value"] == 0.0
    assert "warning" in captured


def test_integer_uses_default_when_no_examples(captured):
    app.render_field("x", _integer_schema(default=2))
    assert captured["number_input"]["value"] == 2
    assert "warning" not in captured


def test_integer_falls_back_and_warns_when_metadata_missing(captured):
    app.render_field("x", _integer_schema())
    assert captured["number_input"]["value"] == 0
    assert "warning" in captured


def test_string_uses_default_when_no_examples(captured):
    app.render_field("x", {"type": "string", "default": "gelu"})
    assert captured["text_input"]["value"] == "gelu"
    assert "warning" not in captured


def test_string_falls_back_to_empty_when_metadata_missing(captured):
    app.render_field("x", {"type": "string"})
    assert captured["text_input"]["value"] == ""
    assert "warning" in captured
