"""Tkinter front end over :mod:`animation_converter.service`."""

from __future__ import annotations

import copy
import json
import queue
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .models import ConversionMode, SourceFormat
from .packaging import PackageMode
from .service import ConversionRequest, ConverterService

DISPLAY_COLLECTION_LIMIT = 200
DISPLAY_STRING_LIMIT = 4000


def _compact_for_display(value, depth: int = 0):
    if depth >= 8:
        return "[additional nested data omitted from GUI; see the saved JSON report]"
    if isinstance(value, dict):
        items = list(value.items())
        compacted = {
            str(key): _compact_for_display(item, depth + 1)
            for key, item in items[:DISPLAY_COLLECTION_LIMIT]
        }
        if len(items) > DISPLAY_COLLECTION_LIMIT:
            compacted["_guiDisplayNote"] = (
                f"{len(items) - DISPLAY_COLLECTION_LIMIT} additional fields omitted; see the saved JSON report."
            )
        return compacted
    if isinstance(value, list):
        compacted = [_compact_for_display(item, depth + 1) for item in value[:DISPLAY_COLLECTION_LIMIT]]
        if len(value) > DISPLAY_COLLECTION_LIMIT:
            compacted.append(
                f"[{len(value) - DISPLAY_COLLECTION_LIMIT} additional items omitted; see the saved JSON report]"
            )
        return compacted
    if isinstance(value, str) and len(value) > DISPLAY_STRING_LIMIT:
        return value[:DISPLAY_STRING_LIMIT] + "... [truncated in GUI]"
    return value


def _display_json(value) -> str:
    return json.dumps(_compact_for_display(value), ensure_ascii=False, indent=2, sort_keys=True)


class AnimationConverterGUI:
    def __init__(self, root: Tk, service: ConverterService) -> None:
        self.root = root
        self.service = service
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_requested = threading.Event()
        self.worker: threading.Thread | None = None
        self.pending_conversion: ConversionRequest | None = None
        self.pending_preview_report: dict | None = None

        root.title("Animation Converter 2.0")
        root.minsize(820, 620)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill=BOTH, expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)

        self.source = StringVar()
        self.output = StringVar()
        self.source_format = StringVar(value=SourceFormat.AUTO.value)
        self.target_format = StringVar(value=SourceFormat.OSTIM_SA.value)
        self.mode = StringVar(value=ConversionMode.NORMAL.value)
        self.behavior = StringVar(value="none")
        self.copy_assets = BooleanVar(value=True)
        self.status = StringVar(value="Idle")

        ttk.Label(frame, text="Source").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.source).grid(row=0, column=1, sticky="ew", padx=8)
        source_buttons = ttk.Frame(frame)
        source_buttons.grid(row=0, column=2)
        ttk.Button(source_buttons, text="Archive", command=self._browse_archive).pack(side=LEFT)
        ttk.Button(source_buttons, text="Folder", command=self._browse_folder).pack(side=LEFT, padx=(4, 0))

        ttk.Label(frame, text="Source format").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.source_format,
            values=[item.value for item in SourceFormat],
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Inspect", command=self.inspect).grid(row=1, column=2, sticky="ew")

        ttk.Label(frame, text="Target format").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.target_format,
            values=[SourceFormat.OSTIM_SA.value, SourceFormat.OSA_OSEX.value],
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=8)

        ttk.Label(frame, text="Output").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.output).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Choose", command=self._browse_output).grid(row=3, column=2, sticky="ew")

        ttk.Label(frame, text="Mode").grid(row=4, column=0, sticky="w", pady=4)
        mode_frame = ttk.Frame(frame)
        mode_frame.grid(row=4, column=1, sticky="w", padx=8)
        for value, label in (
            (ConversionMode.NORMAL.value, "Normal"),
            (ConversionMode.STRICT.value, "Strict"),
            (ConversionMode.BEST_EFFORT.value, "Best effort"),
            (ConversionMode.SALVAGE.value, "Salvage"),
        ):
            ttk.Radiobutton(mode_frame, text=label, value=value, variable=self.mode).pack(side=LEFT, padx=(0, 10))

        ttk.Label(frame, text="Behavior").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.behavior,
            values=["none", "pandora", "nemesis"],
            state="readonly",
        ).grid(row=5, column=1, sticky="ew", padx=8)
        ttk.Checkbutton(frame, text="Copy HKX assets", variable=self.copy_assets).grid(row=5, column=2, sticky="w")

        command_frame = ttk.Frame(frame)
        command_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        self.convert_button = ttk.Button(command_frame, text="Preview and Convert", command=self.convert)
        self.convert_button.pack(side=LEFT)
        self.cancel_button = ttk.Button(command_frame, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side=LEFT, padx=6)
        ttk.Label(command_frame, textvariable=self.status).pack(side=RIGHT)

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.diagnostics = ScrolledText(frame, wrap="word", height=22)
        self.diagnostics.grid(row=8, column=0, columnspan=3, sticky="nsew")
        root.after(100, self._poll_messages)

    def _browse_archive(self) -> None:
        value = filedialog.askopenfilename(filetypes=[("Animation archives", "*.zip *.7z *.rar"), ("All files", "*.*")])
        if value:
            self.source.set(value)

    def _browse_folder(self) -> None:
        value = filedialog.askdirectory()
        if value:
            self.source.set(value)

    def _browse_output(self) -> None:
        value = filedialog.askdirectory()
        if value:
            self.output.set(value)

    def _begin(self, operation) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.cancel_requested.clear()
        self.convert_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start(12)
        self.status.set("Working")
        self.worker = threading.Thread(target=operation, daemon=True)
        self.worker.start()

    def inspect(self) -> None:
        source = Path(self.source.get()).expanduser()
        source_format = SourceFormat(self.source_format.get())

        def work() -> None:
            try:
                result = self.service.inspect(source, source_format)
                self.messages.put(("inspection", result.to_dict()))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
            finally:
                self.messages.put(("done", None))

        self._begin(work)

    def convert(self) -> None:
        source = Path(self.source.get()).expanduser()
        output_text = self.output.get().strip()
        if not output_text:
            messagebox.showerror("Output required", "Choose an output directory or ZIP path.")
            return
        output = Path(output_text).expanduser()
        request = ConversionRequest(
            input_path=source,
            source_format=SourceFormat(self.source_format.get()),
            target_format=SourceFormat(self.target_format.get()),
            output_path=output,
            mode=ConversionMode(self.mode.get()),
            package_mode=PackageMode.ZIP if output.suffix.casefold() == ".zip" else PackageMode.DIRECTORY,
            behavior=self.behavior.get(),
            copy_assets=self.copy_assets.get(),
            cancel_event=self.cancel_requested,
        )

        preview_request = copy.copy(request)
        preview_request.dry_run = True
        preview_request.report_path = None

        def work() -> None:
            try:
                if self.cancel_requested.is_set():
                    return
                result = self.service.convert(preview_request)
                self.messages.put(("preview", (result.report, request)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
            finally:
                self.messages.put(("done", None))

        self._begin(work)

    def _run_conversion(self, request: ConversionRequest) -> None:
        def work() -> None:
            try:
                if self.cancel_requested.is_set():
                    return
                result = self.service.convert(request)
                report = dict(result.report)
                if result.package_result:
                    report["outputPath"] = str(result.package_result.path)
                    report["reportLocation"] = (
                        "conversion-report.json inside the output ZIP"
                        if result.package_result.mode == PackageMode.ZIP
                        else str(result.package_result.path / "conversion-report.json")
                    )
                self.messages.put(("conversion", report))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
            finally:
                self.messages.put(("done", None))

        self._begin(work)

    def cancel(self) -> None:
        self.cancel_requested.set()
        self.status.set("Cancellation requested")

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind in {"inspection", "conversion"}:
                    self.diagnostics.delete("1.0", END)
                    self.diagnostics.insert(END, _display_json(value))
                    if kind == "conversion":
                        readiness = value.get("installReadiness", {}) if isinstance(value, dict) else {}
                        quality = value.get("conversionQuality", "failed") if isinstance(value, dict) else "failed"
                        self.status.set("Converted" if readiness.get("installReady") else f"Converted: {quality}")
                    else:
                        self.status.set("Inspection complete")
                elif kind == "preview":
                    report, request = value
                    self.diagnostics.delete("1.0", END)
                    self.diagnostics.insert(END, _display_json(report))
                    self.status.set(f"Preview: {report.get('conversionQuality', 'unknown')}")
                    self.pending_conversion = request
                    self.pending_preview_report = report
                elif kind == "error":
                    self.status.set("Failed")
                    messagebox.showerror("Animation Converter", str(value))
                elif kind == "done":
                    self.worker = None
                    self.progress.stop()
                    self.convert_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    if self.pending_conversion is not None:
                        pending = self.pending_conversion
                        preview_report = self.pending_preview_report or {}
                        self.pending_conversion = None
                        self.pending_preview_report = None
                        diagnostic_counts = preview_report.get("diagnosticCounts", {})
                        blocking = diagnostic_counts.get("FATAL", 0) or (
                            diagnostic_counts.get("ERROR", 0)
                            and pending.mode in {ConversionMode.NORMAL, ConversionMode.STRICT}
                        )
                        if blocking:
                            messagebox.showerror(
                                "Conversion blocked",
                                "The preview contains blocking diagnostics. Review them before packaging.",
                            )
                        elif messagebox.askyesno(
                            "Package conversion",
                            "Review the diagnostics and losses shown above. Continue with packaging?",
                        ):
                            self.root.after(0, lambda request=pending: self._run_conversion(request))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)


def launch_gui(service: ConverterService | None = None) -> None:
    root = Tk()
    AnimationConverterGUI(root, service or ConverterService())
    root.mainloop()
