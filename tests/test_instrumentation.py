import torch
from torch import nn

from functional_emotions.instrumentation import (
    capture_layer_output,
    decoder_layers,
    resolve_layer_index,
    steer_layer_output,
)


class TupleBlock(nn.Module):
    def forward(self, hidden):
        return (hidden * 2, "cache")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TupleBlock(), TupleBlock()])


def test_decoder_layer_discovery_and_resolution():
    model = TinyModel()
    assert len(decoder_layers(model)) == 2
    assert resolve_layer_index("middle", 2) == 1
    assert resolve_layer_index(-1, 2) == 1


def test_capture_and_zero_steering_are_exact():
    layer = TupleBlock()
    hidden = torch.ones(1, 3, 4)
    baseline = layer(hidden)[0]
    direction = torch.arange(4, dtype=torch.float32)

    with capture_layer_output(layer) as captured:
        observed = layer(hidden)[0]
    assert torch.equal(observed, baseline)
    assert torch.equal(captured[0], baseline)

    with steer_layer_output(layer, direction, 0.0):
        steered = layer(hidden)[0]
    assert torch.equal(steered, baseline)


def test_nonzero_steering_changes_selected_positions_only():
    layer = TupleBlock()
    hidden = torch.zeros(1, 3, 4)
    direction = torch.ones(4)
    with steer_layer_output(layer, direction, 0.5, token_positions=slice(1, 2)):
        steered = layer(hidden)[0]
    assert torch.equal(steered[:, 0], torch.zeros(1, 4))
    assert torch.equal(steered[:, 1], torch.full((1, 4), 0.5))
    assert torch.equal(steered[:, 2], torch.zeros(1, 4))
