#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/common.py

Shared analysis primitives used by both classic and Young modes:
  - _gaussian            : 1-D Gaussian function
  - envelope_center      : Smoothed envelope-based image center detection
  - detect_orientation_fft : 2D FFT fringe orientation estimation
  - fit_envelope         : Gaussian envelope fit on a 1-D profile
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def _gaussian(x, A, x0, sigma, offset):
    """1-D Gaussian: A * exp(-(x-x0)^2 / (2*sigma^2)) + offset."""
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset


def envelope_center(img):
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


def detect_orientation_fft(img):
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


def fit_envelope(profile):
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
