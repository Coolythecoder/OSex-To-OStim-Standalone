"""Behavior-generator writer plugins."""

from .base import BehaviorWriter, BehaviorWriteResult, run_external_generator
from .nemesis import NemesisBehaviorWriter
from .pandora import NoBehaviorWriter, PandoraBehaviorWriter


def behavior_writer(name: str) -> BehaviorWriter:
    normalized = name.casefold()
    if normalized == "pandora":
        return PandoraBehaviorWriter()
    if normalized == "nemesis":
        return NemesisBehaviorWriter()
    if normalized == "none":
        return NoBehaviorWriter()
    raise ValueError(f"Unknown behavior writer: {name}")


__all__ = [
    "BehaviorWriteResult",
    "BehaviorWriter",
    "NemesisBehaviorWriter",
    "NoBehaviorWriter",
    "PandoraBehaviorWriter",
    "behavior_writer",
    "run_external_generator",
]
