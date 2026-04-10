#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/classic.py

Classic diffraction mode analysis:
  - compute_profile_and_minima : extracts horizontal intensity profile,
                                  detects primary diffraction minima,
                                  and fits a Gaussian envelope.
"""

import numpy as np
from scipy.signal import argrelextrema

from analysis.common import fit_envelope


def compute_profile_and_minima(gray, x0, y0, cut_h):
    """
    Compute horizontal intensity profiles and detect primary diffraction
    minima, plus a Gaussian envelope fit.

    Parameters
    ----------
    gray : ndarray
        2-D float grayscale image array.
    x0 : float
        X-coordinate of the envelope center.
    y0 : float
        Y-coordinate of the envelope center.
    cut_h : int
        Height of the vertical averaging band (pixels).

    Returns
    -------
    dict with keys:
        profile_mean        : ndarray or None
        profile_median      : ndarray or None
        left_min            : int or None
        right_min           : int or None
        size_px             : int or None
        left_ripples        : list of int
        right_ripples       : list of int
        classic_fit_params  : tuple or None  — (A, x0, sigma, offset)
        classic_airy_size   : float or None  — envelope FWHM in pixels
        cut_bbox            : tuple or None  — (x, y, w, h) of averaging band
    """
    result = dict(
        profile_mean=None,
        profile_median=None,
        left_min=None,
        right_min=None,
        size_px=None,
        left_ripples=[],
        right_ripples=[],
        classic_fit_params=None,
        classic_airy_size=None,
        cut_bbox=None,
    )

    rows, cols = gray.shape
    half_down = (cut_h - 1) // 2
    half_up = cut_h - half_down - 1
    y_min = max(0, int(round(y0)) - half_down)
    y_max = min(rows, int(round(y0)) + half_up + 1)

    cut_block = gray[y_min:y_max, :]
    if cut_block.size == 0:
        return result

    profile_mean = cut_block.mean(axis=0)
    profile_median = np.median(cut_block, axis=0)
    result["profile_mean"] = profile_mean
    result["profile_median"] = profile_median
    result["cut_bbox"] = (0, y_min, cols, y_max - y_min)

    # Gaussian envelope fit
    fit_params, airy_size = fit_envelope(profile_mean)
    result["classic_fit_params"] = fit_params
    result["classic_airy_size"] = airy_size

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
    result["left_ripples"] = left_minima_all.tolist()
    result["right_ripples"] = right_minima_all.tolist()

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

    left_min = find_primary_min(
        range(peak_idx - 1, -1, -1),
        lambda lb, rb: int(min(rb, peak_idx - 1)),
    )
    if left_min is None and left_minima_all.size > 0:
        left_min = int(left_minima_all[-1])
    if left_min is None and peak_idx > 0:
        left_min = int(np.argmin(profile_mean[:peak_idx]))
    result["left_min"] = left_min

    right_min = find_primary_min(
        range(peak_idx + 1, L),
        lambda lb, rb: int(max(lb, peak_idx + 1)),
    )
    if right_min is None and right_minima_all.size > 0:
        right_min = int(right_minima_all[0])
    if right_min is None and peak_idx < L - 1:
        seg = profile_mean[peak_idx + 1:]
        if seg.size > 0:
            right_min = int(np.argmin(seg)) + peak_idx + 1
    result["right_min"] = right_min

    if left_min is not None and right_min is not None:
        result["size_px"] = right_min - left_min

    return result
