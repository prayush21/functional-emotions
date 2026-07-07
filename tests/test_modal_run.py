from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloud"))

from prototype_registry import (  # noqa: E402
    artifact_output_dir,
    resolve_spec,
    resolve_stage,
    volume_subdir,
)


@pytest.mark.parametrize(
    ("value", "expected_module"),
    [
        ("0", "functional_emotions.prototype0"),
        ("1", "functional_emotions.prototype1"),
        ("2", "functional_emotions.prototype2"),
        ("25", "functional_emotions.prototype25"),
        ("2.5", "functional_emotions.prototype25"),
        ("3", "functional_emotions.prototype3"),
        ("4", "functional_emotions.prototype4"),
        ("5", "functional_emotions.prototype5"),
        ("51", "functional_emotions.prototype51"),
        ("5.1", "functional_emotions.prototype51"),
        ("prototype51", "functional_emotions.prototype51"),
    ],
)
def test_resolve_spec_maps_forms(value: str, expected_module: str) -> None:
    assert resolve_spec(value).module == expected_module


def test_resolve_spec_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_spec("99")


def test_stage_defaults_for_prototype1() -> None:
    spec = resolve_spec("1")
    assert resolve_stage(spec, None) == "extract"
    assert resolve_stage(spec, "all") == "all"


def test_stage_defaults_for_prototype25() -> None:
    spec = resolve_spec("25")
    assert resolve_stage(spec, None) == "prepare"
    assert resolve_stage(spec, "validate") == "validate"
    assert spec.takes_run_dir is True


def test_stage_rejected_for_stageless_prototype() -> None:
    spec = resolve_spec("51")
    assert resolve_stage(spec, None) is None
    with pytest.raises(ValueError):
        resolve_stage(spec, "extract")


def test_invalid_stage_rejected() -> None:
    spec = resolve_spec("1")
    with pytest.raises(ValueError):
        resolve_stage(spec, "nope")


@pytest.mark.parametrize(
    ("output_dir", "expected"),
    [
        ("artifacts/prototype51", "/artifacts/prototype51"),
        ("artifacts/prototype0-qwen3-1.7b-colab", "/artifacts/prototype0-qwen3-1.7b-colab"),
        ("artifacts/prototype25", "/artifacts/prototype25"),
    ],
)
def test_artifact_output_dir(output_dir: str, expected: str) -> None:
    assert artifact_output_dir({"output_dir": output_dir}) == expected


def test_artifact_output_dir_custom_root() -> None:
    assert artifact_output_dir({"output_dir": "artifacts/prototype3"}, "/vol") == "/vol/prototype3"


def test_volume_subdir() -> None:
    assert volume_subdir("51") == "prototype51"
    assert volume_subdir("2.5") == "prototype25"
