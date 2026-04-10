#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/young.py

Young double-slit diffraction mode analysis:
  - analyze_young : FFT-based orientation detection, image rotation,
                    interfringe spacing and Airy envelope size measurement.
"""

import numpy as np
import cv2
from scipy.signal import savgol_filter, find_peaks

from analysis.common import detect_orientation_fft, envelope_center, fit_envelope


def analyze_young(gray):
    """
    Analyze a Young double-slit diffraction pattern.

    Steps
    -----
    1. Detect fringe orientation via 2D FFT.
    2. Rotate image to align fringes vertically.
    3. Compute 2D envelope center on the rotated image.
    4. Extract horizontal profile (mean across rows).
    5. Measure interfringe spacing via peak detection.
    6. Estimate Airy envelope size via Gaussian fit.

    Parameters
    ----------
    gray : ndarray
        2-D float grayscale image array.

    Returns
    -------
    dict with keys:
        rotated         : ndarray   — rotation-aligned image
        fft_mag         : ndarray   — log-magnitude FFT (for visualization)
        angle_deg       : float     — rotation angle applied
        center          : (float, float) — (cx, cy) envelope center
        profile         : ndarray   — raw horizontal mean profile
        smooth          : ndarray   — Savitzky-Golay smoothed profile
        peaks           : ndarray   — fringe peak indices
        interfringe     : float or None
        fit_params      : tuple or None — (A, x0, sigma, offset)
        airy_size       : float or None — envelope FWHM in pixels
        left_airy       : int or None
        right_airy      : int or None
    """
    img = gray.copy()
    h, w = img.shape

    # 1. FFT orientation
    angle_deg, fft_mag = detect_orientation_fft(img)

    # 2. Rotate
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    # 3. Envelope-based center on rotated image
    cx, cy = envelope_center(rotated)

    # 4. Profile
    profile = np.mean(rotated, axis=0)
    win = min(51, len(profile) if len(profile) % 2 == 1 else len(profile) - 1)
    win = max(win, 5)
    smooth = savgol_filter(profile, win, 3)

    # 5. Interfringe
    peaks, _ = find_peaks(smooth, distance=10)
    interfringe = float(np.mean(np.diff(peaks))) if len(peaks) > 1 else None

    # 6. Envelope fit
    fit_params, airy_size = fit_envelope(profile)

    # Airy boundaries from fit
    left_airy = right_airy = None
    if fit_params is not None:
        _, x0_fit, sigma_fit, _ = fit_params
        half_fwhm = 1.1775 * abs(sigma_fit)
        left_airy = int(round(x0_fit - half_fwhm))
        right_airy = int(round(x0_fit + half_fwhm))

    return dict(
        rotated=rotated,
        fft_mag=fft_mag,
        angle_deg=angle_deg,
        center=(cx, cy),
        profile=profile,
        smooth=smooth,
        peaks=peaks,
        interfringe=interfringe,
        fit_params=fit_params,
        airy_size=airy_size,
        left_airy=left_airy,
        right_airy=right_airy,
    )
