import yaml

from src.config_io import load_config

REFERENCE_CONFIG = "configs/mechanistic_hybrid_v24_reference.yaml"
INFERENCE_CONFIG = "inference/mechanistic_hybrid_v24.yaml"


def test_reference_config_is_standalone_and_uses_shared_data() -> None:
    with open(REFERENCE_CONFIG) as handle:
        raw = yaml.safe_load(handle)
    config = load_config(REFERENCE_CONFIG)

    assert "extends" not in raw
    assert config.model.architecture == "mechanistic_hybrid_v24"
    assert config.data.root_dir.startswith("/network/projects/amlrt/")
    assert config.data.root_dir.endswith("data_samples_v6_native_context_512_crop_256_ignition_probability_mass_log_firesize")
    assert config.logger.enabled is False
    assert config.training.max_epochs == 40
    assert config.data.batch_size * config.training.gradient_accumulation_steps == 64
    assert config.evaluation.best_ckpt_metrics == ["hex/mean/ccc"]


def test_reference_inference_config_uses_published_artifacts() -> None:
    with open(INFERENCE_CONFIG) as handle:
        config = yaml.safe_load(handle)

    assert config["checkpoint_path"].startswith("/network/projects/amlrt/")
    assert config["checkpoint_path"].endswith("mechanistic_hybrid_v24_native_512_crop_256_firesize_q3/best.pth")
    assert config["training_data_root"].startswith("/network/projects/amlrt/")
    assert config["training_data_root"].endswith("data_samples_v6_native_context_512_crop_256_ignition_probability_mass_log_firesize")
    assert config["prepare_data"] is True
    assert config["weather_norm_params_path"] is None
    assert config["fire_size_norm_params_path"] is None
