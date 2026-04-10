#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots/young_plots.py

Matplotlib figures for Young double-slit diffraction mode:
  - show_young_plots : rotated image, FFT spectrum, annotated profile.
"""

import numpy as np
import matplotlib.pyplot as plt

from analysis.common import _gaussian


def show_young_plots(result):
    """
    Display three diagnostic figures for Young mode.

    Figures
    -------
    1. "Young — Rotated Image"
       Rotation-corrected image with profile axis, fringe markers,
       and Airy boundary lines.
    2. "Young — FFT Spectrum"
       Log-magnitude FFT with detected orientation angle.
    3. "Young — Profile"
       Raw and smoothed horizontal profiles, fringe peak markers,
       Gaussian envelope fit, and Airy boundary lines.

    Parameters
    ----------
    result : dict
        Output dictionary from analysis.young.analyze_young(), containing:
          rotated, fft_mag, angle_deg, center, profile, smooth,
          peaks, interfringe, fit_params, airy_size, left_airy, right_airy.
    """
    rotated    = result["rotated"]
    fft_mag    = result["fft_mag"]
    angle_deg  = result["angle_deg"]
    cx, cy     = result["center"]
    profile    = result["profile"]
    smooth     = result["smooth"]
    peaks      = result["peaks"]
    interfringe = result["interfringe"]
    fit_params = result["fit_params"]
    airy_size  = result["airy_size"]
    left_airy  = result["left_airy"]
    right_airy = result["right_airy"]

    # --- Figure 1: rotated image ---
    plt.figure("Young — Rotated Image")
    plt.clf()
    plt.imshow(rotated, cmap="gray", origin="upper")
    plt.axhline(cy, color="yellow", linestyle="--", lw=1, label="Profile axis")
    if peaks is not None and len(peaks) > 0:
        plt.scatter(
            peaks, [cy] * len(peaks),
            color="red", s=20, zorder=5, label="Fringes",
        )
    if left_airy is not None:
        plt.axvline(left_airy, color="lime", linestyle=":", lw=1.5, label="Airy boundary")
        plt.axvline(right_airy, color="lime", linestyle=":", lw=1.5)
    plt.title(f"Young — Rotated (FFT angle ≈ {angle_deg:.1f}°)")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.legend(loc="upper right")

    # --- Figure 2: FFT spectrum ---
    plt.figure("Young — FFT Spectrum")
    plt.clf()
    plt.imshow(fft_mag, cmap="inferno", origin="upper")
    plt.title(f"FFT log-magnitude  |  detected angle = {angle_deg:.2f}°")
    plt.colorbar(label="log(1 + |F|)")
    plt.xlabel("u (frequency)")
    plt.ylabel("v (frequency)")

    # --- Figure 3: profile ---
    plt.figure("Young — Profile")
    plt.clf()
    x = np.arange(len(profile))
    plt.plot(x, profile, color="steelblue", lw=1.0, alpha=0.6, label="Raw profile")
    plt.plot(x, smooth, color="blue", lw=1.5, label="Smoothed profile")

    if peaks is not None and len(peaks) > 0:
        plt.scatter(
            peaks, smooth[peaks],
            color="red", zorder=6, label=f"Fringes (n={len(peaks)})",
        )

    if fit_params is not None:
        plt.plot(x, _gaussian(x, *fit_params), "k--", lw=1.5, label="Gaussian envelope (est.)")

    if left_airy is not None:
        plt.axvline(left_airy, color="lime", linestyle=":", lw=1.2)
        plt.axvline(right_airy, color="lime", linestyle=":", lw=1.2)

    title_parts = ["Young — Profile"]
    if interfringe is not None:
        title_parts.append(f"Interfringe ≈ {interfringe:.2f} px")
    if airy_size is not None:
        title_parts.append(f"Envelope FWHM ≈ {airy_size:.2f} px")
    plt.title("\n".join(title_parts))
    plt.xlabel("x (pixels)")
    plt.ylabel("Intensity")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.ion()
    plt.show()
