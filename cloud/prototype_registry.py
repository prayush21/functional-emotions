"""Prototype dispatch metadata shared by the Modal runner and its tests.

This module deliberately imports nothing from ``modal`` so the mapping logic can
be exercised locally (and in CI) without a Modal token or the ``modal`` package.
The Modal entrypoint lives in ``cloud/modal_run.py`` and reuses these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrototypeSpec:
    """Describes how to launch one experiment prototype."""

    key: str
    module: str
    stages: tuple[str, ...] = ()
    default_stage: str | None = None
    takes_run_dir: bool = False
    aliases: tuple[str, ...] = field(default_factory=tuple)


PROTOTYPE_SPECS: tuple[PrototypeSpec, ...] = (
    PrototypeSpec(key="0", module="functional_emotions.prototype0"),
    PrototypeSpec(
        key="1",
        module="functional_emotions.prototype1",
        stages=("generate", "extract", "all"),
        default_stage="extract",
    ),
    PrototypeSpec(key="2", module="functional_emotions.prototype2"),
    PrototypeSpec(
        key="25",
        module="functional_emotions.prototype25",
        stages=("prepare", "generate", "extract", "validate", "all"),
        default_stage="prepare",
        takes_run_dir=True,
        aliases=("2.5",),
    ),
    PrototypeSpec(key="3", module="functional_emotions.prototype3"),
    PrototypeSpec(key="4", module="functional_emotions.prototype4"),
    PrototypeSpec(key="5", module="functional_emotions.prototype5"),
    PrototypeSpec(key="51", module="functional_emotions.prototype51", aliases=("5.1",)),
)


def _normalize(prototype: str) -> str:
    return str(prototype).strip().lower().removeprefix("prototype").strip(".")


def resolve_spec(prototype: str) -> PrototypeSpec:
    """Map a CLI ``--prototype`` value to its :class:`PrototypeSpec`."""

    normalized = _normalize(prototype)
    for spec in PROTOTYPE_SPECS:
        if normalized == spec.key or normalized in {_normalize(a) for a in spec.aliases}:
            return spec
    known = ", ".join(spec.key for spec in PROTOTYPE_SPECS)
    raise ValueError(f"Unknown prototype {prototype!r}; expected one of: {known}")


def resolve_stage(spec: PrototypeSpec, stage: str | None) -> str | None:
    """Validate a requested stage against a spec, falling back to its default."""

    if not spec.stages:
        if stage is not None:
            raise ValueError(f"Prototype {spec.key} does not accept a --stage argument")
        return None
    chosen = stage if stage is not None else spec.default_stage
    if chosen not in spec.stages:
        allowed = ", ".join(spec.stages)
        raise ValueError(f"Prototype {spec.key} stage must be one of: {allowed}")
    return chosen


def artifact_output_dir(config: dict, volume_root: str = "/artifacts") -> str:
    """Rewrite a config's ``output_dir`` onto the artifacts volume.

    Preserves the per-prototype subdirectory: ``artifacts/prototype51`` becomes
    ``/artifacts/prototype51`` so downloaded bundles keep their expected layout.
    """

    original = str(config.get("output_dir", "")).strip()
    leaf = original.rsplit("/", 1)[-1] if original else "prototype"
    leaf = leaf.removeprefix("artifacts/") or "prototype"
    return f"{volume_root.rstrip('/')}/{leaf}"


def volume_subdir(prototype: str) -> str:
    """Local/remote artifacts subdirectory name for a prototype, e.g. ``prototype51``."""

    return f"prototype{resolve_spec(prototype).key}"
