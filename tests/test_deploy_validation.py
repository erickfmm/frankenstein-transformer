"""Deploy validation must build task-appropriate inputs for every model class.

Regression gate for experiment 2's 104 failed deploy runs: every ``vit_*``
combo (102 failures) died with ``Input type (long int) and bias type (float)
should be the same`` because :meth:`ModelDeployer.validate_deployment` fed
token ids to the patch-embedding Conv2d, and the raw-EEG audio model (2
failures) received a mono ``(2, 16000)`` waveform where ``feature: raw`` with
``in_channels=8`` requires ``(B, T, C)``. The validator now dispatches on the
recorded model class and training task.
"""
from __future__ import annotations

import pytest
import torch

from src.deploy.deploy import ModelDeployer
from src.model.config import FrankensteinModelConfig
from src.model.frankenstein_audio import FrankensteinAudio
from src.model.frankenstein_encoder import FrankensteinEncoder
from src.model.frankenstein_vit import FrankensteinViT


def _text_config():
    return FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 100, "hidden_size": 64, "num_layers": 1, "num_loops": 1, "num_heads": 4, "num_kv_heads": 2, "dropout": 0.0, "layer_pattern": ['standard'], "mode": 'encoder'}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "norm": {"type": 'layer_norm'}, "bitnet": {"enabled": False}, "mixture_of_depths": {"enabled": False}, "ffn": {"hidden_size": 64, "activation": 'gelu'}, "model_class": 'frankenstein_text_encoder'})


def _vit_config(task):
    return FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 100, "hidden_size": 64, "num_layers": 1, "num_loops": 1, "num_heads": 4, "num_kv_heads": 2, "dropout": 0.0, "layer_pattern": ['standard'], "mode": 'encoder'}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "norm": {"type": 'layer_norm'}, "bitnet": {"enabled": False}, "mixture_of_depths": {"enabled": False}, "ffn": {"hidden_size": 64, "activation": 'gelu'}, "model_class": 'frankenstein_vit', "image": {"height": 32, "width": 32, "patch_size": 16, "in_channels": 3, "num_seg_classes": 2, "mask_ratio": 0.5, "prediction_target": 'mean_color_3bit', "cls_token": True, "pos_embedding_type": 'learned_1d'}, "num_classes": 2, "pooling_mode": 'cls'})


def _audio_config(feature="raw", asr_loss="ctc"):
    return FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 32, "num_layers": 1, "num_loops": 1, "num_heads": 4, "num_kv_heads": 2, "dropout": 0.0, "layer_pattern": ['standard'], "mode": 'encoder'}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "norm": {"type": 'layer_norm'}, "bitnet": {"enabled": False}, "mixture_of_depths": {"enabled": False}, "ffn": {"hidden_size": 32, "activation": 'gelu'}, "model_class": 'frankenstein_audio', "audio": {"feature": feature, "in_channels": 8, "sample_rate": 8000, "conv_channels": 16, "asr_loss": asr_loss}, "num_classes": 3})


# ---------------------------------------------------------------------------
# Unit: the validation input builder
# ---------------------------------------------------------------------------
def test_validation_input_text_encoder_is_token_ids():
    deployer = ModelDeployer(_text_config())
    test_input, kwargs = deployer._build_validation_input()
    assert test_input.dtype == torch.long
    assert test_input.shape == (2, 64)
    assert kwargs == {}


def test_validation_input_ssog_uses_the_configured_grid():
    """SSOG requires seq_len == grid_h * grid_w; the default 64 would crash."""
    config = _text_config()
    config.dims.layer_pattern = ["ssog"]
    config.attention.setdefault("ssog", {})["grid_h"] = 1
    config.attention.setdefault("ssog", {})["grid_w"] = 32
    deployer = ModelDeployer(config)
    test_input, kwargs = deployer._build_validation_input()
    assert test_input.shape == (2, 32)


@pytest.mark.parametrize("task", ["classification", "segmentation", "patch_prediction"])
def test_validation_input_vit_is_float_pixels(task):
    config = _vit_config(task)
    deployer = ModelDeployer(config, task=task)
    test_input, kwargs = deployer._build_validation_input()
    assert test_input.dtype.is_floating_point
    assert test_input.shape == (2, 3, 32, 32)
    assert kwargs == {"task": task}


def test_validation_input_vit_defaults_to_classification():
    deployer = ModelDeployer(_vit_config("classification"))
    test_input, kwargs = deployer._build_validation_input()
    assert kwargs == {"task": "classification"}
    assert test_input.shape == (2, 3, 32, 32)


def test_validation_input_audio_raw_is_multichannel():
    deployer = ModelDeployer(_audio_config(feature="raw"))
    test_input, kwargs = deployer._build_validation_input()
    assert test_input.shape == (2, 8000, 8)
    assert kwargs == {"task": "audio_classification"}


def test_validation_input_audio_logmel_is_mono():
    deployer = ModelDeployer(_audio_config(feature="logmel"))
    test_input, kwargs = deployer._build_validation_input()
    assert test_input.shape == (2, 8000)


def test_validation_input_audio_transducer_gets_target_ids():
    deployer = ModelDeployer(
        _audio_config(feature="logmel", asr_loss="transducer"), task="asr"
    )
    test_input, kwargs = deployer._build_validation_input()
    assert kwargs["task"] == "asr"
    assert kwargs["target_ids"].dtype == torch.long
    assert kwargs["target_ids"].shape[0] == 2


# ---------------------------------------------------------------------------
# Integration: convert + validate round trip
# ---------------------------------------------------------------------------
def _convert_and_validate(config, model, task, tmp_path):
    deployer = ModelDeployer(config, task=task)
    deployer.model = model
    deployer.convert_to_deployment(str(tmp_path), save_format="standard")
    return deployer.validate_deployment(str(tmp_path))


def test_deploy_and_validate_text_encoder(tmp_path):
    config = _text_config()
    assert _convert_and_validate(config, FrankensteinEncoder(config), "mlm", tmp_path)


@pytest.mark.parametrize("task", ["classification", "segmentation", "patch_prediction"])
def test_deploy_and_validate_vit(task, tmp_path):
    config = _vit_config(task)
    assert _convert_and_validate(config, FrankensteinViT(config), task, tmp_path)


def test_deploy_and_validate_vit_without_explicit_task(tmp_path):
    """Old checkpoints carry no task: the ViT default (classification) must work."""
    config = _vit_config("classification")
    assert _convert_and_validate(config, FrankensteinViT(config), None, tmp_path)


def test_deploy_and_validate_audio_raw(tmp_path):
    config = _audio_config(feature="raw")
    assert _convert_and_validate(
        config, FrankensteinAudio(config), "audio_classification", tmp_path
    )


def test_validate_deployment_reads_training_task_from_checkpoint(tmp_path):
    """`training_task` recorded by the trainer drives validation when --task is absent."""
    config = _vit_config("segmentation")
    model = FrankensteinViT(config)
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_jsonable(),
            "model_class": "FrankensteinViT",
            "training_task": "segmentation",
        },
        checkpoint_path,
    )
    deployer = ModelDeployer(_vit_config("segmentation"))
    deployer.load_training_checkpoint(str(checkpoint_path))
    assert deployer.task == "segmentation"
    deployer.convert_to_deployment(str(tmp_path / "deploy"), save_format="standard")
    assert deployer.validate_deployment(str(tmp_path / "deploy"))
