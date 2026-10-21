import torch

from inference.predictor import BurnRiskPredictor


class ConstantModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, spatial_inputs, auxiliary_inputs=None):
        return torch.full((spatial_inputs.shape[0], 1, spatial_inputs.shape[2], spatial_inputs.shape[3]), self.value)


class MultiConstantModel(torch.nn.Module):
    def forward(self, spatial_inputs, auxiliary_inputs=None):
        values = torch.tensor([0.0, 2.0, 3.0], dtype=spatial_inputs.dtype, device=spatial_inputs.device)
        return values.view(1, 3, 1, 1).expand(spatial_inputs.shape[0], 3, spatial_inputs.shape[2], spatial_inputs.shape[3])


def test_predictor_keeps_fi_outputs_linear():
    config = {"data": {"input_sources": [{"name": "grid", "params": {"target_name": "fi"}}]}}
    predictor = BurnRiskPredictor(model=ConstantModel(2.0), device="cpu", config=config)

    predictions = predictor(torch.zeros(1, 1, 2, 2))

    assert torch.all(predictions == 2.0)


def test_predictor_sigmoids_bp_outputs():
    config = {"data": {"input_sources": [{"name": "grid", "params": {"target_name": "bp"}}]}}
    predictor = BurnRiskPredictor(model=ConstantModel(0.0), device="cpu", config=config)

    predictions = predictor(torch.zeros(1, 1, 2, 2))

    assert torch.allclose(predictions, torch.full((1, 1, 2, 2), 0.5))


def test_predictor_activates_and_splits_multi_target_outputs():
    config = {
        "data": {
            "input_sources": [
                {
                    "name": "grid",
                    "params": {
                        "targets": [
                            {"name": "bp", "out_norm": "min_max"},
                            {"name": "fi", "out_norm": "log_standard"},
                            {"name": "ros", "out_norm": "log_standard"},
                        ]
                    },
                }
            ]
        }
    }
    predictor = BurnRiskPredictor(model=MultiConstantModel(), device="cpu", config=config)

    predictions = predictor(torch.zeros(1, 1, 2, 2))
    named = predictor.predict_named_batch(torch.zeros(1, 1, 2, 2))

    assert torch.allclose(predictions[:, 0], torch.full((1, 2, 2), 0.5))
    assert torch.all(predictions[:, 1] == 2.0)
    assert torch.all(predictions[:, 2] == 3.0)
    assert list(named) == ["bp", "fi", "ros"]
    assert torch.equal(named["fi"], predictions[:, 1:2])
