from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn


def decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Return decoder blocks for common Hugging Face causal language models."""
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        current: Any = model
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None and len(current) > 0:
            return current
    raise ValueError(
        "Could not locate decoder layers. Supported layouts: model.layers, "
        "transformer.h, and gpt_neox.layers."
    )


def resolve_layer_index(layer: int | str, number_of_layers: int) -> int:
    if layer == "middle":
        return number_of_layers // 2
    index = int(layer)
    if index < 0:
        index += number_of_layers
    if not 0 <= index < number_of_layers:
        raise IndexError(f"Layer {layer!r} is outside a {number_of_layers}-layer model")
    return index


def _hidden_from_output(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


def _replace_hidden(output: Any, hidden: Tensor) -> Any:
    if isinstance(output, Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


@contextmanager
def capture_layer_output(layer: nn.Module):
    captured: list[Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured.append(_hidden_from_output(output).detach())

    handle = layer.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def steer_layer_output(
    layer: nn.Module,
    direction: Tensor,
    strength: float,
    token_positions: slice | Tensor | None = None,
):
    """Add a direction to selected token positions in a decoder block's output."""
    positions = token_positions if token_positions is not None else slice(None)

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = _hidden_from_output(output)
        modified = hidden.clone()
        delta = direction.to(device=hidden.device, dtype=hidden.dtype) * strength
        modified[:, positions, :] = modified[:, positions, :] + delta
        return _replace_hidden(output, modified)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
