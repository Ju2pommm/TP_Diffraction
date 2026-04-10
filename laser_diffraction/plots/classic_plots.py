#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots/classic_plots.py

Matplotlib figures for classic diffraction mode:
  - show_classic_plots : original image, high-contrast image,
                         annotated intensity profile with Gaussian envelope fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from analysis.common import _gaussian


def show_classic_plots(gray, max_coord, profile_mean, profile_median,
                       left_min, right_min, size_px,
                       classic_fit_params, classic_airy_size,
                       cut_bbox):
    """
    Display three diagnostic figures for classic mode.

    Figures
    -------
    1. "Diffraction Image"
       Original grayscale image with envelope center marker and cut band.
    2. "Diffraction Image (High Contrast)"
       Log + gamma stretch to reveal low-intensity details.
    3. "Horizontal Profile"
       Mean/median curves, detected primary minima, Gaussian envelope fit,
       and size annotations.

    Parameters
    ----------
    gray               : ndarray         — 2-D float grayscale image
    max_coord          : (float, float)  — (cx, cy) envelope center
    profile_mean       : ndarray         — horizontal mean profile
    profile_median     : ndarray         — horizontal median profile
    left_min           : int or None
    right_min          : int or None
    size_px            : int or None
    classic_fit_params : tuple or None   — Gaussian fit (A, x0, sigma, offset)
    classic_airy_size  : float or None   — envelope FWHM
    cut_bbox           : tuple or None   — (x, y, w, h)
    """
    if gray is None or profile_mean is None:
        return

    # --- Figure 1: original image ---
    plt.figure("Diffraction Image")
    plt.clf()
    plt.imshow(gray, cmap="gray", origin="upper")
    if max_coord is not None:
        mx, my = max_coord
        plt.plot(mx, my, "rx", markersize=8, label="Envelope center")
    if cut_bbox is not None:
        bx, by, bw, bh = cut_bbox
        rect = Rectangle(
            (bx, by), bw, bh,
            facecolor="none", edgecolor="yellow",
            linewidth=1.5, linestyle="--", alpha=0.9,
        )
        plt.gca().add_patch(rect)
    title = "Image"
    if max_coord is not None:
        title += f" — Center x={max_coord[0]:.1f}, y={max_coord[1]:.1f}"
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.legend(loc="upper right")

    # --- Figure 2: high-contrast ---
    plt.figure("Diffraction Image (High Contrast)")
    plt.clf()
    p_low, p_high = np.percentile(gray, (1.0, 99.0))
    if p_high <= p_low:
        p_low, p_high = gray.min(), gray.max()
    norm = np.clip((gray - p_low) / (p_high - p_low), 0.0, 1.0)
    log_scaled = np.log1p(norm * 200.0) / np.log1p(200.0)
    stretched = np.clip(np.power(log_scaled, 0.35), 0.0, 1.0)
    plt.imshow(stretched, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    if max_coord is not None:
        plt.plot(max_coord[0], max_coord[1], "rx", markersize=8)
    if cut_bbox is not None:
        bx, by, bw, bh = cut_bbox
        rect = Rectangle(
            (bx, by), bw, bh,
            facecolor="none", edgecolor="cyan",
            linewidth=1.0, linestyle="-", alpha=0.9,
        )
        plt.gca().add_patch(rect)
    title = "High-contrast view"
    if max_coord is not None:
        title += f" — Center x={max_coord[0]:.1f}, y={max_coord[1]:.1f}"
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")

    # --- Figure 3: intensity profile ---
    plt.figure("Horizontal Profile")
    plt.clf()
    x = np.arange(profile_mean.size)
    plt.plot(x, profile_mean, color="blue", lw=1.5, label="Mean (cut)")
    plt.plot(x, profile_median, color="magenta", lw=1.0, linestyle="--", label="Median (cut)")

    if max_coord is not None:
        plt.axvline(int(round(max_coord[0])), color="red", linestyle="--", label="Peak x")

    if left_min is not None:
        plt.scatter(
            [left_min], [profile_mean[left_min]],
            color="green", zorder=6, label="Left primary minimum",
        )
        plt.axvline(left_min, color="green", linestyle=":", lw=1.0)

    if right_min is not None:
        plt.scatter(
            [right_min], [profile_mean[right_min]],
            color="orange", zorder=6, label="Right primary minimum",
        )
        plt.axvline(right_min, color="orange", linestyle=":", lw=1.0)

    if classic_fit_params is not None:
        fit_curve = _gaussian(x, *classic_fit_params)
        plt.plot(x, fit_curve, "k--", lw=1.2, label="Gaussian envelope (est.)")

    if classic_airy_size is not None:
        plt.text(
            len(x) // 2,
            profile_mean.max() * 0.60,
            f"Envelope FWHM ≈ {classic_airy_size:.1f} px",
            ha="center",
            bbox=dict(facecolor="white", alpha=0.75),
        )

    if size_px is not None and left_min is not None and right_min is not None:
        mid = (left_min + right_min) / 2.0
        plt.text(
            mid,
            profile_mean.max() * 0.80,
            f"Size (minima): {size_px} px",
            ha="center",
            bbox=dict(facecolor="lightyellow", alpha=0.85),
        )

    title = "Horizontal intensity profile"
    if max_coord is not None:
        title += f" — Peak x={max_coord[0]:.1f}, y={max_coord[1]:.1f}"
    plt.title(title)
    plt.xlabel("x (pixels)")
    plt.ylabel("Intensity")
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.ion()
    plt.show()
