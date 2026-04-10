#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:05:54 2026

@author: julien
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laser_diffraction_viewer.py

Tkinter application to analyze laser diffraction images.

Features
--------
Classic mode:
  - Load an image and detect the envelope center (smoothed, not raw brightest pixel)
  - Compute a horizontal intensity profile (mean and median) over a user-defined cut
  - Detect primary minima around the central peak using a plateau-aware method
  - Fit a Gaussian envelope to estimate Airy disk size (FWHM)
  - Display: original image, high-contrast image, annotated profile with fit

Young double-slit mode:
  - Detect fringe orientation via 2D FFT (robust, replaces gradient method)
  - Rotate image to align fringes vertically
  - Extract horizontal profile and fit Gaussian envelope
  - Measure interfringe spacing (peak-to-peak) and Airy envelope size (FWHM)
  - Display: rotated image, FFT spectrum, annotated profile

Common:
  - Measurement history log in GUI
  - Save results per measurement
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image
import cv2
from scipy.signal import savgol_filter, find_peaks, argrelextrema
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(x, A, x0, sigma, offset):
    """1-D Gaussian: A * exp(-(x-x0)^2 / (2*sigma^2)) + offset."""
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class DiffractionViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laser Diffraction Viewer")
        self.geometry("560x200")
        self._build_ui()
        self._reset_state()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.image = None
        self.image_path = None
        self.gray = None
        self.max_coord = None          # (x, y) envelope center
        self.profile_mean = None
        self.profile_median = None
        self.left_min = None
        self.right_min = None
        self.size_px = None
        self._current_cut_bbox = None
        self.left_ripples = []
        self.right_ripples = []
        self.history = []
        # Classic mode fit
        self.classic_fit_params = None
        self.classic_airy_size = None
        # Young mode results
        self.young_interfringe = None
        self.young_airy_size = None

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        ctrl = tk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        tk.Button(ctrl, text="Open Image...", command=self.open_image).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Analyze & Show", command=self.analyze).pack(side=tk.LEFT, padx=6)

        tk.Label(ctrl, text="Cut height (px):").pack(side=tk.LEFT, padx=(12, 2))
        self.cut_spin = tk.Spinbox(ctrl, from_=1, to=1000, width=6)
        self.cut_spin.delete(0, "end")
        self.cut_spin.insert(0, "10")
        self.cut_spin.pack(side=tk.LEFT)

        tk.Button(ctrl, text="Update Cut", command=self.update_cut).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Save Size...", command=self.save_size).pack(side=tk.LEFT, padx=6)

        tk.Label(ctrl, text="Mode:").pack(side=tk.LEFT, padx=(12, 2))
        self.mode_var = tk.StringVar(value="classic")
        tk.OptionMenu(ctrl, self.mode_var, "classic", "young").pack(side=tk.LEFT)

        tk.Button(ctrl, text="Quit", command=self.quit).pack(side=tk.RIGHT)

        self.info_var = tk.StringVar(value="Open an image to begin.")
        tk.Label(self, textvariable=self.info_var, anchor="w").pack(
            side=tk.TOP, fill=tk.X, padx=6
        )

        history_frame = tk.Frame(self)
        history_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)
        tk.Label(history_frame, text="Measurement History:").pack(anchor="w")
        self.history_text = tk.Text(history_frame, height=6)
        self.history_text.pack(fill=tk.BOTH, expand=True)

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

        mode = self.mode_var.get()
        if mode == "classic":
            center = self._envelope_center(self.gray)
            self.max_coord = center
            try:
                cut_h = max(1, int(self.cut_spin.get()))
            except Exception:
                cut_h = 10
            self._compute_profile_and_minima(center[0], center[1], cut_h)
            self._show_plots()
            self._update_info_classic()
        else:
            self._analyze_young()

    def update_cut(self):
        if self.gray is None:
            messagebox.showwarning("Update Cut", "No image loaded.")
            return
        if self.max_coord is None:
            messagebox.showwarning("Update Cut", "Run Analyze first.")
            return
        try:
            cut_h = max(1, int(self.cut_spin.get()))
        except Exception:
            cut_h = 10
        x0, y0 = self.max_coord
        self._compute_profile_and_minima(x0, y0, cut_h)
        self._show_plots()
        self._update_info_classic()

    # ------------------------------------------------------------------
    # Envelope-based center detection (shared by both modes)
    # ------------------------------------------------------------------

    def _envelope_center(self, img):
        """
        Return (cx, cy) as the center of the intensity envelope.

        Uses a heavily smoothed profile on each axis rather than the
        raw brightest pixel, which is sensitive to noise and hot pixels.
        """
        rows, cols = img.shape
        win_y = min(201, rows if rows % 2 == 1 else rows - 1)
        win_x = min(201, cols if cols % 2 == 1 else cols - 1)
        win_y = max(win_y, 5)
        win_x = max(win_x, 5)

        vprofile = savgol_filter(img.mean(axis=1), win_y, 3)
        hprofile = savgol_filter(img.mean(axis=0), win_x, 3)
        cy = float(np.argmax(vprofile))
        cx = float(np.argmax(hprofile))
        return cx, cy

    # ------------------------------------------------------------------
    # FFT orientation detection
    # ------------------------------------------------------------------

    def _detect_orientation_fft(self, img):
        """
        Estimate dominant fringe orientation using 2D FFT.

        Diffraction fringes are periodic → strong off-axis peaks in the
        Fourier magnitude spectrum. The angle of the strongest peak
        (relative to DC) gives the fringe orientation.

        Returns
        -------
        angle_deg : float
            Rotation angle (degrees) to align fringes vertically.
        fft_mag : ndarray
            Log-magnitude FFT array (for visualization).
        """
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Suppress DC component
        dc_r = max(5, min(h, w) // 20)
        magnitude[cy - dc_r: cy + dc_r, cx - dc_r: cx + dc_r] = 0

        # Guard against degenerate (uniform) images
        if magnitude.max() == 0:
            return 0.0, magnitude

        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        dy = peak_idx[0] - cy
        dx = peak_idx[1] - cx

        # Fringe direction is perpendicular to the frequency vector
        fringe_angle = np.degrees(np.arctan2(dy, dx))
        return fringe_angle, magnitude

    # ------------------------------------------------------------------
    # Gaussian envelope fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_envelope(profile):
        """
        Fit a Gaussian envelope to a 1-D intensity profile.

        This approximates the Airy disk envelope. Results should be
        interpreted as estimates; a Gaussian is not an exact Airy function.

        Returns
        -------
        fit_params : tuple or None
            (A, x0, sigma, offset)
        fwhm : float or None
            Full width at half maximum — used as Airy size estimate.
        """
        x = np.arange(len(profile))
        A0 = float(np.max(profile))
        x0_0 = float(np.argmax(profile))
        sigma0 = len(profile) / 10.0
        offset0 = float(np.min(profile))

        try:
            popt, _ = curve_fit(
                _gaussian, x, profile,
                p0=[A0, x0_0, sigma0, offset0],
                maxfev=10000,
            )
        except Exception:
            return None, None

        A, x0, sigma, offset = popt

        # Sanity checks — reject nonsensical fits
        if sigma <= 0 or abs(sigma) > len(profile) / 2:
            return None, None
        if A <= 0:
            return None, None

        fwhm = 2.3548 * abs(sigma)
        return popt, fwhm

    # ------------------------------------------------------------------
    # Classic mode — profile + minima
    # ------------------------------------------------------------------

    def _compute_profile_and_minima(self, x0, y0, cut_h):
        """
        Compute horizontal intensity profiles and detect primary diffraction
        minima, plus a Gaussian envelope fit.

        Stores results in instance attributes.
        """
        rows, cols = self.gray.shape
        half_down = (cut_h - 1) // 2
        half_up = cut_h - half_down - 1
        y_min = max(0, int(round(y0)) - half_down)
        y_max = min(rows, int(round(y0)) + half_up + 1)

        cut_block = self.gray[y_min:y_max, :]
        if cut_block.size == 0:
            self.profile_mean = None
            self.profile_median = None
            self.left_min = self.right_min = self.size_px = None
            self.left_ripples = self.right_ripples = []
            self.classic_fit_params = self.classic_airy_size = None
            return

        profile_mean = cut_block.mean(axis=0)
        profile_median = np.median(cut_block, axis=0)
        self.profile_mean = profile_mean
        self.profile_median = profile_median

        # Gaussian envelope fit
        self.classic_fit_params, self.classic_airy_size = self._fit_envelope(profile_mean)

        peak_idx = int(round(x0))
        L = profile_mean.size

        minima = argrelextrema(profile_mean, np.less, order=3)[0]
        if minima.size == 0:
            fallbacks = []
            if peak_idx > 0:
                fallbacks.append(int(np.argmin(profile_mean[:peak_idx])))
            if peak_idx < L - 1:
                fallbacks.append(
                    int(np.argmin(profile_mean[peak_idx + 1:])) + peak_idx + 1
                )
            minima = np.array(fallbacks, dtype=int)

        left_minima_all = minima[minima < peak_idx]
        right_minima_all = minima[minima > peak_idx]
        self.left_ripples = left_minima_all.tolist()
        self.right_ripples = right_minima_all.tolist()

        eps = 1e-8

        def plateau_bounds(i):
            val = profile_mean[i]
            lb = i
            while lb - 1 >= 0 and abs(profile_mean[lb - 1] - val) <= eps:
                lb -= 1
            rb = i
            while rb + 1 < L and abs(profile_mean[rb + 1] - val) <= eps:
                rb += 1
            return lb, rb, val

        def find_primary_min(scan_range, clamp_fn):
            for i in scan_range:
                val = profile_mean[i]
                if not (0.0 <= val <= 3.0):
                    continue
                lb, rb, v = plateau_bounds(i)
                lo = lb - 1
                ro = rb + 1
                lo_ok = (lo < 0) or (profile_mean[lo] >= v - eps)
                ro_ok = (ro >= L) or (profile_mean[ro] >= v - eps)
                lo_strict = (lo >= 0) and (profile_mean[lo] > v + eps)
                ro_strict = (ro < L) and (profile_mean[ro] > v + eps)
                if lo_ok and ro_ok and (lo_strict or ro_strict):
                    return clamp_fn(lb, rb)
            return None

        self.left_min = find_primary_min(
            range(peak_idx - 1, -1, -1),
            lambda lb, rb: int(min(rb, peak_idx - 1)),
        )
        if self.left_min is None and left_minima_all.size > 0:
            self.left_min = int(left_minima_all[-1])
        if self.left_min is None and peak_idx > 0:
            self.left_min = int(np.argmin(profile_mean[:peak_idx]))

        self.right_min = find_primary_min(
            range(peak_idx + 1, L),
            lambda lb, rb: int(max(lb, peak_idx + 1)),
        )
        if self.right_min is None and right_minima_all.size > 0:
            self.right_min = int(right_minima_all[0])
        if self.right_min is None and peak_idx < L - 1:
            seg = profile_mean[peak_idx + 1:]
            if seg.size > 0:
                self.right_min = int(np.argmin(seg)) + peak_idx + 1

        self.size_px = (
            self.right_min - self.left_min
            if self.left_min is not None and self.right_min is not None
            else None
        )
        self._current_cut_bbox = (0, y_min, cols, y_max - y_min)

    # ------------------------------------------------------------------
    # Classic mode — plots
    # ------------------------------------------------------------------

    def _show_plots(self):
        """
        Display three diagnostic figures for classic mode:

        1. Original grayscale image with envelope center marker and cut region.
        2. High-contrast (log + gamma stretch) version.
        3. Horizontal intensity profile with mean/median, detected minima,
           and Gaussian envelope fit (Airy size estimate).
        """
        if self.gray is None or self.profile_mean is None:
            return

        # Figure 1 — original image
        plt.figure("Diffraction Image")
        plt.clf()
        plt.imshow(self.gray, cmap="gray", origin="upper")
        if self.max_coord is not None:
            mx, my = self.max_coord
            plt.plot(mx, my, "rx", markersize=8, label="Envelope center")
        if self._current_cut_bbox is not None:
            bx, by, bw, bh = self._current_cut_bbox
            rect = Rectangle(
                (bx, by), bw, bh,
                facecolor="none", edgecolor="yellow",
                linewidth=1.5, linestyle="--", alpha=0.9,
            )
            plt.gca().add_patch(rect)
        title = "Image"
        if self.max_coord is not None:
            title += f" — Center x={self.max_coord[0]:.1f}, y={self.max_coord[1]:.1f}"
        plt.title(title)
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.legend(loc="upper right")

        # Figure 2 — high-contrast
        plt.figure("Diffraction Image (High Contrast)")
        plt.clf()
        p_low, p_high = np.percentile(self.gray, (1.0, 99.0))
        if p_high <= p_low:
            p_low, p_high = self.gray.min(), self.gray.max()
        norm = np.clip((self.gray - p_low) / (p_high - p_low), 0.0, 1.0)
        log_scaled = np.log1p(norm * 200.0) / np.log1p(200.0)
        stretched = np.clip(np.power(log_scaled, 0.35), 0.0, 1.0)
        plt.imshow(stretched, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
        if self.max_coord is not None:
            plt.plot(self.max_coord[0], self.max_coord[1], "rx", markersize=8)
        if self._current_cut_bbox is not None:
            bx, by, bw, bh = self._current_cut_bbox
            rect = Rectangle(
                (bx, by), bw, bh,
                facecolor="none", edgecolor="cyan",
                linewidth=1.0, linestyle="-", alpha=0.9,
            )
            plt.gca().add_patch(rect)
        title = "High-contrast view"
        if self.max_coord is not None:
            title += f" — Center x={self.max_coord[0]:.1f}, y={self.max_coord[1]:.1f}"
        plt.title(title)
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")

        # Figure 3 — intensity profile
        plt.figure("Horizontal Profile")
        plt.clf()
        x = np.arange(self.profile_mean.size)
        plt.plot(x, self.profile_mean, color="blue", lw=1.5, label="Mean (cut)")
        plt.plot(
            x, self.profile_median,
            color="magenta", lw=1.0, linestyle="--", label="Median (cut)",
        )
        if self.max_coord is not None:
            plt.axvline(int(round(self.max_coord[0])), color="red", linestyle="--", label="Peak x")
        if self.left_min is not None:
            plt.scatter(
                [self.left_min], [self.profile_mean[self.left_min]],
                color="green", zorder=6, label="Left primary minimum",
            )
            plt.axvline(self.left_min, color="green", linestyle=":", lw=1.0)
        if self.right_min is not None:
            plt.scatter(
                [self.right_min], [self.profile_mean[self.right_min]],
                color="orange", zorder=6, label="Right primary minimum",
            )
            plt.axvline(self.right_min, color="orange", linestyle=":", lw=1.0)

        # Gaussian envelope fit
        if self.classic_fit_params is not None:
            fit_curve = _gaussian(x, *self.classic_fit_params)
            plt.plot(x, fit_curve, "k--", lw=1.2, label="Gaussian envelope (est.)")
        if self.classic_airy_size is not None:
            plt.text(
                len(x) // 2,
                self.profile_mean.max() * 0.60,
                f"Envelope FWHM ≈ {self.classic_airy_size:.1f} px",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.75),
            )

        if self.size_px is not None and self.left_min is not None and self.right_min is not None:
            mid = (self.left_min + self.right_min) / 2.0
            plt.text(
                mid,
                self.profile_mean.max() * 0.80,
                f"Size (minima): {self.size_px} px",
                ha="center",
                bbox=dict(facecolor="lightyellow", alpha=0.85),
            )

        title = "Horizontal intensity profile"
        if self.max_coord is not None:
            title += f" — Peak x={self.max_coord[0]:.1f}, y={self.max_coord[1]:.1f}"
        plt.title(title)
        plt.xlabel("x (pixels)")
        plt.ylabel("Intensity")
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.ion()
        plt.show()

    # ------------------------------------------------------------------
    # Young mode
    # ------------------------------------------------------------------

    def _analyze_young(self):
        """
        Analyze a Young double-slit diffraction pattern.

        Steps
        -----
        1. Detect fringe orientation via 2D FFT (robust to noise/misalignment).
        2. Rotate image to align fringes vertically.
        3. Compute 2D envelope center on the rotated image.
        4. Extract horizontal profile (mean across rows near center).
        5. Measure interfringe spacing via peak detection.
        6. Estimate Airy envelope size via Gaussian fit.

        Figures
        -------
        - "Young — Rotated Image": rotated image with fringes, center, Airy boundaries.
        - "Young — FFT Spectrum": log-magnitude FFT with detected orientation.
        - "Young — Profile": smoothed profile, fringe peaks, envelope fit.
        """
        img = self.gray.copy()
        h, w = img.shape

        # 1. FFT orientation
        angle_deg, fft_mag = self._detect_orientation_fft(img)

        # 2. Rotate
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

        # 3. Envelope-based center on rotated image
        cx, cy = self._envelope_center(rotated)
        self.max_coord = (cx, cy)

        # 4. Profile (horizontal mean across image)
        profile = np.mean(rotated, axis=0)
        win = min(51, len(profile) if len(profile) % 2 == 1 else len(profile) - 1)
        win = max(win, 5)
        smooth = savgol_filter(profile, win, 3)

        # 5. Interfringe
        peaks, _ = find_peaks(smooth, distance=10)
        interfringe = float(np.mean(np.diff(peaks))) if len(peaks) > 1 else None

        # 6. Envelope fit
        fit_params, airy_size = self._fit_envelope(profile)

        # Airy boundaries from fit
        left_airy = right_airy = None
        if fit_params is not None:
            _, x0_fit, sigma_fit, _ = fit_params
            half_fwhm = 1.1775 * abs(sigma_fit)  # half of FWHM
            left_airy = int(round(x0_fit - half_fwhm))
            right_airy = int(round(x0_fit + half_fwhm))

        self.young_interfringe = interfringe
        self.young_airy_size = airy_size

        # --- Figure 1: Rotated image ---
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

        # --- Figure 3: Profile ---
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

        # Update GUI
        info = "Young mode — "
        if interfringe is not None:
            info += f"Interfringe ≈ {interfringe:.2f} px"
        if airy_size is not None:
            info += f"  |  Airy FWHM ≈ {airy_size:.1f} px"
        self.info_var.set(info)

    # ------------------------------------------------------------------
    # Info bar helpers
    # ------------------------------------------------------------------

    def _update_info_classic(self):
        info = ""
        if self.max_coord is not None:
            info += f"Center x={self.max_coord[0]:.2f}, y={self.max_coord[1]:.2f}. "
        if self.size_px is not None:
            info += (
                f"Size (minima) ≈ {self.size_px} px "
                f"({self.left_min} → {self.right_min}). "
            )
        if self.classic_airy_size is not None:
            info += f"Envelope FWHM ≈ {self.classic_airy_size:.1f} px."
        if not info:
            info = "Could not determine primary minima."
        self.info_var.set(info)

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


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = DiffractionViewer()
    app.mainloop()