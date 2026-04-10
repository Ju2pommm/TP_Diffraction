#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py

DiffractionViewer — Tkinter application root and analysis orchestrator.

This module is intentionally thin: it owns application state and
coordinates calls between the UI, analysis, and plotting modules.
No analysis logic or figure-building code lives here.

Module responsibilities
-----------------------
ui.py               build_ui()              — widget construction
analysis/common.py  envelope_center()       — shared center detection
analysis/classic.py compute_profile_and_minima() — classic mode analysis
analysis/young.py   analyze_young()         — Young mode analysis
plots/classic_plots.py show_classic_plots() — classic mode figures
plots/young_plots.py   show_young_plots()   — Young mode figures
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image

from ui import build_ui
from analysis.common import envelope_center
from analysis.classic import compute_profile_and_minima
from analysis.young import analyze_young
from plots.classic_plots import show_classic_plots
from plots.young_plots import show_young_plots


class DiffractionViewer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Laser Diffraction Viewer")
        self.geometry("560x200")
        build_ui(self)
        self._reset_state()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.image = None
        self.image_path = None
        self.gray = None
        self.max_coord = None

        # Classic mode results
        self.profile_mean = None
        self.profile_median = None
        self.left_min = None
        self.right_min = None
        self.size_px = None
        self._current_cut_bbox = None
        self.left_ripples = []
        self.right_ripples = []
        self.classic_fit_params = None
        self.classic_airy_size = None

        # Young mode results
        self.young_interfringe = None
        self.young_airy_size = None
        self._young_result = None     # full result dict from analyze_young()

        self.history = []

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.png"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            im = Image.open(path)
        except Exception as e:
            messagebox.showerror("Open Image", f"Failed to open image:\n{e}")
            return

        self._reset_state()
        self.image = im.copy()
        self.image_path = path
        self.gray = np.array(im.convert("L"), dtype=np.float64)
        self.info_var.set(
            f"Loaded {os.path.basename(path)} "
            f"({self.gray.shape[1]}×{self.gray.shape[0]})"
        )

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def analyze(self):
        if self.gray is None:
            messagebox.showwarning("Analyze", "No image loaded.")
            return

        if self.mode_var.get() == "classic":
            self.max_coord = envelope_center(self.gray)
            cut_h = self._read_cut_h()
            self._run_classic(cut_h)
        else:
            self._run_young()

    def update_cut(self):
        if self.gray is None:
            messagebox.showwarning("Update Cut", "No image loaded.")
            return
        if self.max_coord is None:
            messagebox.showwarning("Update Cut", "Run Analyze first.")
            return
        cut_h = self._read_cut_h()
        self._run_classic(cut_h)

    # ------------------------------------------------------------------
    # Classic mode orchestration
    # ------------------------------------------------------------------

    def _run_classic(self, cut_h):
        x0, y0 = self.max_coord
        result = compute_profile_and_minima(self.gray, x0, y0, cut_h)

        # Unpack result dict into instance state
        self.profile_mean       = result["profile_mean"]
        self.profile_median     = result["profile_median"]
        self.left_min           = result["left_min"]
        self.right_min          = result["right_min"]
        self.size_px            = result["size_px"]
        self.left_ripples       = result["left_ripples"]
        self.right_ripples      = result["right_ripples"]
        self.classic_fit_params = result["classic_fit_params"]
        self.classic_airy_size  = result["classic_airy_size"]
        self._current_cut_bbox  = result["cut_bbox"]

        show_classic_plots(
            gray               = self.gray,
            max_coord          = self.max_coord,
            profile_mean       = self.profile_mean,
            profile_median     = self.profile_median,
            left_min           = self.left_min,
            right_min          = self.right_min,
            size_px            = self.size_px,
            classic_fit_params = self.classic_fit_params,
            classic_airy_size  = self.classic_airy_size,
            cut_bbox           = self._current_cut_bbox,
        )
        self._update_info_classic()

    # ------------------------------------------------------------------
    # Young mode orchestration
    # ------------------------------------------------------------------

    def _run_young(self):
        result = analyze_young(self.gray)
        self._young_result = result

        self.max_coord        = result["center"]
        self.young_interfringe = result["interfringe"]
        self.young_airy_size  = result["airy_size"]

        show_young_plots(result)
        self._update_info_young()

    # ------------------------------------------------------------------
    # Info bar helpers
    # ------------------------------------------------------------------

    def _update_info_classic(self):
        parts = []
        if self.max_coord is not None:
            parts.append(f"Center x={self.max_coord[0]:.2f}, y={self.max_coord[1]:.2f}")
        if self.size_px is not None:
            parts.append(
                f"Size (minima) ≈ {self.size_px} px "
                f"({self.left_min} → {self.right_min})"
            )
        if self.classic_airy_size is not None:
            parts.append(f"Envelope FWHM ≈ {self.classic_airy_size:.1f} px")
        self.info_var.set("  |  ".join(parts) if parts else "Could not determine primary minima.")

    def _update_info_young(self):
        parts = ["Young mode"]
        if self.young_interfringe is not None:
            parts.append(f"Interfringe ≈ {self.young_interfringe:.2f} px")
        if self.young_airy_size is not None:
            parts.append(f"Airy FWHM ≈ {self.young_airy_size:.1f} px")
        self.info_var.set("  |  ".join(parts))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_cut_h(self):
        try:
            return max(1, int(self.cut_spin.get()))
        except Exception:
            self.cut_spin.delete(0, "end")
            self.cut_spin.insert(0, "10")
            return 10

    # ------------------------------------------------------------------
    # Save to history
    # ------------------------------------------------------------------

    def save_size(self):
        """
        Append the current measurement to the on-screen history log.

        Classic mode records the minima-based size and envelope FWHM.
        Young mode records interfringe spacing and Airy FWHM.
        """
        mode = self.mode_var.get()

        if mode == "classic":
            if self.size_px is None and self.classic_airy_size is None:
                messagebox.showwarning("Save", "No measurement available.")
                return
            parts = ["[Classic]"]
            if self.size_px is not None:
                parts.append(f"Size={self.size_px} px (minima: {self.left_min}→{self.right_min})")
            if self.classic_airy_size is not None:
                parts.append(f"Env.FWHM={self.classic_airy_size:.1f} px")
            entry = "  ".join(parts)

        else:  # young
            if self.young_interfringe is None and self.young_airy_size is None:
                messagebox.showwarning("Save", "No measurement available.")
                return
            parts = ["[Young]"]
            if self.young_interfringe is not None:
                parts.append(f"Interfringe={self.young_interfringe:.2f} px")
            if self.young_airy_size is not None:
                parts.append(f"Airy FWHM={self.young_airy_size:.1f} px")
            entry = "  ".join(parts)

        if self.image_path:
            entry = f"{os.path.basename(self.image_path)} → {entry}"

        self.history.append(entry)
        self.history_text.insert(tk.END, entry + "\n")
        self.history_text.see(tk.END)
