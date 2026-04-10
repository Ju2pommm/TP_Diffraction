#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui.py

Builds the DiffractionViewer Tkinter interface.

All widget construction is isolated here so that app.py stays focused
on orchestration logic. The function receives the app instance and
attaches all widgets to it, storing references as instance attributes.
"""

import tkinter as tk


def build_ui(app):
    """
    Construct and attach all Tkinter widgets to *app*.

    Widgets stored on *app*
    -----------------------
    app.cut_spin      : Spinbox — vertical cut height
    app.mode_var      : StringVar — "classic" | "young"
    app.info_var      : StringVar — status line text
    app.history_text  : Text widget — measurement log
    """
    ctrl = tk.Frame(app)
    ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

    tk.Button(ctrl, text="Open Image...", command=app.open_image).pack(side=tk.LEFT)
    tk.Button(ctrl, text="Analyze & Show", command=app.analyze).pack(side=tk.LEFT, padx=6)

    tk.Label(ctrl, text="Cut height (px):").pack(side=tk.LEFT, padx=(12, 2))
    app.cut_spin = tk.Spinbox(ctrl, from_=1, to=1000, width=6)
    app.cut_spin.delete(0, "end")
    app.cut_spin.insert(0, "10")
    app.cut_spin.pack(side=tk.LEFT)

    tk.Button(ctrl, text="Update Cut", command=app.update_cut).pack(side=tk.LEFT, padx=6)
    tk.Button(ctrl, text="Save Size...", command=app.save_size).pack(side=tk.LEFT, padx=6)

    tk.Label(ctrl, text="Mode:").pack(side=tk.LEFT, padx=(12, 2))
    app.mode_var = tk.StringVar(value="classic")
    tk.OptionMenu(ctrl, app.mode_var, "classic", "young").pack(side=tk.LEFT)

    tk.Button(ctrl, text="Quit", command=app.quit).pack(side=tk.RIGHT)

    app.info_var = tk.StringVar(value="Open an image to begin.")
    tk.Label(app, textvariable=app.info_var, anchor="w").pack(
        side=tk.TOP, fill=tk.X, padx=6
    )

    history_frame = tk.Frame(app)
    history_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)
    tk.Label(history_frame, text="Measurement History:").pack(anchor="w")
    app.history_text = tk.Text(history_frame, height=6)
    app.history_text.pack(fill=tk.BOTH, expand=True)
