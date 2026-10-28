#!/usr/bin/env python3
"""Streamlit audio wiring tests.

``build_config_from_schema`` must force ``model_class: frankenstein_audio`` for
the audio tasks and render the ``audio:`` / ``audio_dataset:`` blocks such that
the produced YAML validates against the resolved schema.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from jsonschema import Draft202012Validator

from src.streamlit_gui import app
from src.utils.schema_loader import resolve_schema


def _stub_streamlit(task: str) -> mock.MagicMock:
    """Return a Streamlit stub whose selectboxes pick *task* when offered."""
    stub = mock.MagicMock()

    def selectbox(label, options, index=0, **kwargs):
        choices = list(options)
        if task in choices:
            return task
        return choices[index] if choices else None

    stub.selectbox.side_effect = selectbox
    stub.number_input.side_effect = lambda label, value=None, **kwargs: value
    stub.text_input.side_effect = lambda label, value="", **kwargs: value
    stub.checkbox.side_effect = lambda label, value=False, **kwargs: value
    stub.multiselect.side_effect = (
        lambda label, options, default=None, **kwargs: default or []
    )
    stub.text_area.side_effect = lambda label, value="", **kwargs: value
    stub.radio.side_effect = lambda label, options, **kwargs: options[0]
    return stub


@pytest.fixture(scope="module")
def schema():
    return resolve_schema("src/schema.yaml")


@pytest.mark.parametrize("task", ["asr", "audio_classification"])
def test_audio_task_builds_valid_yaml(schema, task):
    with mock.patch.object(app, "st", _stub_streamlit(task)):
        config = app.build_config_from_schema(schema)

    assert config["training"]["task"] == task
    assert config["model_class"] == "frankenstein_audio"
    assert "audio" in config
    assert "audio_dataset" in config

    document = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
    errors = list(Draft202012Validator(schema).iter_errors(document))
    assert not errors, errors[0].message
