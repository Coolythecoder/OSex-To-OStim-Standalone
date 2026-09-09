"""Command-line interface for the shared conversion service."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from .adapters.ostim_sa import schema_summary
from .archive import ArchiveError, ArchiveSecurityError
from .detection import DetectionError
from .models import ConversionMode, SourceFormat
from .packaging import PackageMode, PackagingError
from .service import (
    EXIT_EXTRACTION_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_SUCCESS,
    EXIT_UNSUPPORTED,
    EXIT_VALIDATION_FAILURE,
    EXIT_WARNINGS,
    ConversionCancelled,
    ConversionRequest,
    ConverterService,
    UnsupportedConversionError,
)

FORMAT_CHOICES = [item.value for item in SourceFormat]
TARGET_CHOICES = [SourceFormat.OSTIM_SA.value, SourceFormat.OSA_OSEX.value]


def _format(value: str) -> SourceFormat:
    return SourceFormat(value)


def _mode(args: argparse.Namespace) -> ConversionMode:
    if getattr(args, "strict", False):
        return ConversionMode.STRICT
    if getattr(args, "best_effort", False):
        return ConversionMode.BEST_EFFORT
    if getattr(args, "salvage", False):
        return ConversionMode.SALVAGE
    return ConversionMode.NORMAL


def _add_mode_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--strict", action="store_true", help="Fail on warnings or any recorded loss.")
    group.add_argument(
        "--best-effort", action="store_true", help="Write validated incomplete output with an explicit loss report."
    )
    group.add_argument(
        "--salvage", action="store_true", help="Explicitly allow visibly incomplete HKX asset placeholders."
    )


def _add_source_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from", dest="source_format", choices=FORMAT_CHOICES, default="auto")


def _add_conversion_options(parser: argparse.ArgumentParser) -> None:
    _add_source_option(parser)
    parser.add_argument("--to", dest="target_format", choices=TARGET_CHOICES, required=True)
    parser.add_argument("--output", type=Path)
    _add_mode_options(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--package", choices=[item.value for item in PackageMode], default="directory")
    parser.add_argument("--behavior", choices=["pandora", "nemesis", "none"], default="none")
    parser.add_argument("--pack-id")
    parser.add_argument("--display-name")
    parser.add_argument("--copy-assets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roundtrip-sidecar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m animation_converter",
        description="Bidirectional Skyrim animation-pack converter with explicit loss accounting.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show tracebacks for unexpected failures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Detect and summarize an input without writing output.")
    inspect_parser.add_argument("input", type=Path)
    _add_source_option(inspect_parser)

    validate_parser = subparsers.add_parser("validate", help="Parse and validate an input without conversion.")
    validate_parser.add_argument("input", type=Path)
    _add_source_option(validate_parser)
    validate_parser.add_argument("--json", action="store_true", dest="json_output")

    convert_parser = subparsers.add_parser("convert", help="Convert an archive, directory, or metadata file.")
    convert_parser.add_argument("input", type=Path)
    _add_conversion_options(convert_parser)

    roundtrip_parser = subparsers.add_parser(
        "roundtrip", help="Convert through one format and back, then compare semantics."
    )
    roundtrip_parser.add_argument("input", type=Path)
    _add_source_option(roundtrip_parser)
    roundtrip_parser.add_argument("--through", choices=TARGET_CHOICES, required=True)
    roundtrip_parser.add_argument("--output", type=Path, required=True)
    _add_mode_options(roundtrip_parser)
    roundtrip_parser.add_argument("--package", choices=[item.value for item in PackageMode], default="directory")
    roundtrip_parser.add_argument("--behavior", choices=["pandora", "nemesis", "none"], default="none")
    roundtrip_parser.add_argument("--pack-id")
    roundtrip_parser.add_argument("--display-name")
    roundtrip_parser.add_argument("--copy-assets", action=argparse.BooleanOptionalAction, default=True)
    roundtrip_parser.add_argument("--overwrite", action="store_true")

    schema_parser = subparsers.add_parser("schema-dump", help="Print pinned schema fields or adapter capabilities.")
    schema_parser.add_argument("format", choices=[item.value for item in SourceFormat if item != SourceFormat.AUTO])

    subparsers.add_parser("gui", help="Launch the Tkinter interface.")
    return parser


def _request_from_args(args: argparse.Namespace) -> ConversionRequest:
    if not args.dry_run and args.output is None:
        raise ValueError("convert requires --output unless --dry-run is selected")
    return ConversionRequest(
        input_path=args.input,
        target_format=_format(args.target_format),
        output_path=args.output,
        source_format=_format(args.source_format),
        mode=_mode(args),
        package_mode=PackageMode(args.package),
        behavior=args.behavior,
        pack_id=args.pack_id,
        display_name=args.display_name,
        copy_assets=args.copy_assets,
        dry_run=args.dry_run,
        report_path=args.report,
        roundtrip_sidecar=args.roundtrip_sidecar,
        overwrite=args.overwrite,
        deterministic=not args.non_deterministic,
    )


def _print_diagnostics(diagnostics) -> None:
    if not diagnostics:
        print("No diagnostics.")
        return
    for item in diagnostics:
        location = f" [{item.source_file}]" if item.source_file else ""
        print(f"{item.severity.name} {item.code}{location}: {item.message}")
        if item.remediation:
            print(f"  Suggested: {item.remediation}")


def _validation_exit(diagnostics) -> int:
    if diagnostics.has_errors:
        return EXIT_VALIDATION_FAILURE
    if diagnostics.has_warnings:
        return EXIT_WARNINGS
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = ConverterService()
    try:
        if args.command == "inspect":
            result = service.inspect(args.input, _format(args.source_format))
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            if result.selected.proposed_format == SourceFormat.AUTO:
                return EXIT_UNSUPPORTED
            return _validation_exit(result.ir.diagnostics if result.ir else [])

        if args.command == "validate":
            ir, diagnostics = service.validate(args.input, _format(args.source_format))
            if args.json_output:
                print(
                    json.dumps(
                        {
                            "sourceFormat": ir.source.format.value,
                            "sceneCount": len(ir.graph.nodes),
                            "diagnosticCounts": diagnostics.counts(),
                            "diagnostics": diagnostics.to_list(),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Detected {ir.source.format.value}; parsed {len(ir.graph.nodes)} scene(s).")
                _print_diagnostics(diagnostics)
            return _validation_exit(diagnostics)

        if args.command == "convert":
            result = service.convert(_request_from_args(args))
            print(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True))
            if result.package_result:
                print(f"Output: {result.package_result.path}", file=sys.stderr)
            return result.exit_code

        if args.command == "roundtrip":
            inspection = service.inspect(args.input, _format(args.source_format))
            if inspection.selected.proposed_format == SourceFormat.AUTO:
                raise DetectionError("Could not detect the source format for round-trip conversion.")
            request = ConversionRequest(
                input_path=args.input,
                target_format=_format(args.through),
                output_path=args.output,
                source_format=_format(args.source_format),
                mode=_mode(args),
                package_mode=PackageMode(args.package),
                behavior=args.behavior,
                pack_id=args.pack_id,
                display_name=args.display_name,
                copy_assets=args.copy_assets,
                overwrite=args.overwrite,
            )
            result = service.roundtrip(request, through=_format(args.through))
            print(
                json.dumps(
                    {
                        "sourceDigest": result.source_digest,
                        "roundtripDigest": result.roundtrip_digest,
                        "semanticallyEqual": result.semantically_equal,
                        "firstExitCode": result.first.exit_code,
                        "secondExitCode": result.second.exit_code if result.second else None,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            if result.second is None or not result.second.output_written:
                return EXIT_VALIDATION_FAILURE
            return EXIT_SUCCESS if result.semantically_equal else EXIT_WARNINGS

        if args.command == "schema-dump":
            source_format = _format(args.format)
            if source_format == SourceFormat.OSTIM_SA:
                payload = schema_summary()
            else:
                payload = service.registry.get(source_format).capabilities().to_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return EXIT_SUCCESS

        if args.command == "gui":
            from .gui import launch_gui

            launch_gui(service)
            return EXIT_SUCCESS
        parser.error(f"Unknown command: {args.command}")
        return EXIT_INTERNAL_ERROR
    except UnsupportedConversionError as exc:
        print(f"Unsupported conversion: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    except (ArchiveSecurityError, ArchiveError) as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return EXIT_EXTRACTION_FAILURE
    except (ConversionCancelled, DetectionError, PackagingError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILURE
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return EXIT_VALIDATION_FAILURE
    except Exception as exc:
        print(f"Internal error: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
