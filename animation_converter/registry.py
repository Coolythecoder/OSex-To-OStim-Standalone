"""Format adapter registry."""

from __future__ import annotations

from .adapters.base import Adapter
from .models import SourceFormat


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[SourceFormat, Adapter] = {}

    def register(self, adapter: Adapter) -> None:
        if adapter.format in self._adapters:
            raise ValueError(f"An adapter is already registered for {adapter.format.value}")
        self._adapters[adapter.format] = adapter

    def get(self, source_format: SourceFormat) -> Adapter:
        if source_format == SourceFormat.AUTO:
            raise KeyError("AUTO is a detection request, not an adapter")
        try:
            return self._adapters[source_format]
        except KeyError as exc:
            raise KeyError(f"No adapter is registered for {source_format.value}") from exc

    def all(self) -> list[Adapter]:
        return list(self._adapters.values())

    def capabilities(self) -> list[dict[str, object]]:
        return [adapter.capabilities().to_dict() for adapter in self.all()]


def default_registry() -> AdapterRegistry:
    # Imports are intentionally lazy so the model layer remains lightweight.
    from .adapters.flowergirls import FlowerGirlsAdapter
    from .adapters.osa_osex import OsaOsexAdapter
    from .adapters.ostim_legacy import OStimLegacyAdapter
    from .adapters.ostim_sa import OStimSAAdapter
    from .adapters.slal import SlalAdapter

    registry = AdapterRegistry()
    for adapter in (
        OStimSAAdapter(),
        OStimLegacyAdapter(),
        OsaOsexAdapter(),
        SlalAdapter(),
        FlowerGirlsAdapter(),
    ):
        registry.register(adapter)
    return registry
