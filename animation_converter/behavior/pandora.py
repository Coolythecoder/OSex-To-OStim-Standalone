"""Pandora manifest-only writer."""

from __future__ import annotations

import json
from pathlib import Path

from ..diagnostics import DiagnosticCollection
from ..models import ConversionIR
from .base import BehaviorWriteResult


def _registration_rows(ir: ConversionIR) -> list[dict[str, object]]:
    assets = {asset.id: asset for asset in ir.assets}
    rows: list[dict[str, object]] = []
    for event in sorted(ir.events, key=lambda item: item.name.casefold()):
        for actor_index, asset_id in sorted(event.actor_assets.items()):
            asset = assets.get(asset_id)
            rows.append(
                {
                    "animationEvent": event.name,
                    "actorIndex": actor_index,
                    "assetId": asset_id,
                    "hkxPath": asset.target_path if asset else None,
                    "assetHash": asset.sha256 if asset else None,
                }
            )
    return rows


def write_registration_manifest(
    ir: ConversionIR,
    destination: Path,
    pack_id: str,
    generator: str,
) -> Path:
    metadata = destination / "Data" / "SKSE" / "Plugins" / "OStim" / "converter_metadata" / pack_id
    metadata.mkdir(parents=True, exist_ok=True)
    output = metadata / "behavior-registration.json"
    payload = {
        "schema": "animation-converter.behavior-registration/1",
        "generator": generator,
        "registrationMode": "manifest-only",
        "externalGenerationRequired": generator != "none",
        "registrations": _registration_rows(ir),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


class PandoraBehaviorWriter:
    name = "pandora"

    def write(self, ir: ConversionIR, destination: Path, pack_id: str) -> BehaviorWriteResult:
        diagnostics = DiagnosticCollection()
        output = write_registration_manifest(ir, destination, pack_id, self.name)
        diagnostics.warning(
            "PANDORA_EXTERNAL_GENERATION_REQUIRED",
            "A validated event manifest was written, but Pandora patch syntax was not fabricated. Run Pandora explicitly after installation.",
            category="behavior-registration completeness",
            remediation="Install the converted files, run Pandora, inspect its output, and retain the conversion report.",
        )
        return BehaviorWriteResult(self.name, [output], False, True, diagnostics)


class NoBehaviorWriter:
    name = "none"

    def write(self, ir: ConversionIR, destination: Path, pack_id: str) -> BehaviorWriteResult:
        diagnostics = DiagnosticCollection()
        output = write_registration_manifest(ir, destination, pack_id, self.name)
        diagnostics.warning(
            "BEHAVIOR_REGISTRATION_DISABLED",
            "No behavior-generator output was requested; the package cannot be labelled install ready.",
            category="behavior-registration completeness",
            remediation="Choose Pandora or Nemesis and complete the external generation step.",
        )
        return BehaviorWriteResult(self.name, [output], False, False, diagnostics)
