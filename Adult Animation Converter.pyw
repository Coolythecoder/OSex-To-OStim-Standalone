"""Launch the transparent source distribution without opening a console window."""

from __future__ import annotations

import runpy
from pathlib import Path


def show_setup_error(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Adult Animation Converter", message)
        root.destroy()
    except Exception:
        pass


try:
    import customtkinter  # noqa: F401
    from PIL import Image  # noqa: F401
except Exception:
    show_setup_error(
        "The source release needs its two Python GUI dependencies.\n\n"
        "Open Windows Terminal in this folder and run:\n"
        "py -3.11 -m pip install --user -r requirements-runtime.txt"
    )
    raise SystemExit(2)


script = Path(__file__).with_name("Osex-to-OStim-Standalone.py")
if not script.is_file():
    show_setup_error(f"The converter source file is missing:\n{script}")
    raise SystemExit(2)

runpy.run_path(str(script), run_name="__main__")
