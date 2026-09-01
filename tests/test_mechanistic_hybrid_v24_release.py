import yaml

from src.config_io import load_config

REFERENCE_CONFIG = "configs/mechanistic_hybrid_v24_reference.yaml"


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
