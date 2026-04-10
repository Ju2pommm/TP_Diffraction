#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
laser_diffraction_viewer.py

Tkinter application to analyze laser diffraction images.

Features:
- Load an image and detect the brightest point (averaged if multiple pixels share the maximum)
- Compute a horizontal intensity profile (mean and median) over a user-defined vertical cut
- Detect primary minima around the central peak using a plateau-aware method
- Display:
    - Original image with cut region
    - High-contrast enhanced image
    - Intensity profile with detected minima
- Estimate diffraction spot size from the distance between primary minima
- Save results (including ripple positions) to CSV
"""
import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import cv2
try:
    from scipy.signal import argrelextrema
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

class DiffractionViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laser Diffraction Viewer")
        self.geometry("520x140")
        self._build_ui()
        self.image = None
        self.image_path = None
        self.gray = None
        self.max_coord = None  # (x, y) as floats (averaged)
        self.profile_mean = None
        self.profile_median = None
        self.left_min = None
        self.right_min = None
        self.size_px = None
        self._current_cut_bbox = None
        # store ripple coordinates (lists of x indices)
        self.left_ripples = []
        self.right_ripples = []
        self.history = []

    def _build_ui(self):
        ctrl = tk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        btn_open = tk.Button(ctrl, text="Open Image...", command=self.open_image)
        btn_open.pack(side=tk.LEFT)

        btn_analyze = tk.Button(ctrl, text="Analyze & Show", command=self.analyze)
        btn_analyze.pack(side=tk.LEFT, padx=6)

        tk.Label(ctrl, text="Cut height (px):").pack(side=tk.LEFT, padx=(12, 2))
        self.cut_spin = tk.Spinbox(ctrl, from_=1, to=1000, width=6)
        self.cut_spin.delete(0, "end")
        self.cut_spin.insert(0, "10")
        self.cut_spin.pack(side=tk.LEFT)

        btn_update = tk.Button(ctrl, text="Update Cut", command=self.update_cut)
        btn_update.pack(side=tk.LEFT, padx=6)

        btn_save = tk.Button(ctrl, text="Save Size...", command=self.save_size)
        btn_save.pack(side=tk.LEFT, padx=6)

        btn_quit = tk.Button(ctrl, text="Quit", command=self.quit)
        btn_quit.pack(side=tk.RIGHT)

        self.info_var = tk.StringVar(value="Open an image to begin.")
        lbl_info = tk.Label(self, textvariable=self.info_var, anchor="w")
        lbl_info.pack(side=tk.TOP, fill=tk.X, padx=6)
        history_frame = tk.Frame(self)
        history_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        tk.Label(history_frame, text="Measurement History:").pack(anchor="w")

        self.history_text = tk.Text(history_frame, height=6)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        self.mode_var = tk.StringVar(value="classic")

        tk.Label(ctrl, text="Mode:").pack(side=tk.LEFT, padx=(12, 2))

        mode_menu = tk.OptionMenu(ctrl, self.mode_var, "classic", "young")
        mode_menu.pack(side=tk.LEFT)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.png"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            im = Image.open(path)
        except Exception as e:
            messagebox.showerror("Open Image", f"Failed to open image: {e}")
            return

        gray = im.convert("L")
        self.image = im.copy()
        self.image_path = path
        self.gray = np.array(gray, dtype=np.float64)
        self.info_var.set(f"Loaded {os.path.basename(path)} ({self.gray.shape[1]}x{self.gray.shape[0]})")
        # reset measurements
        self.max_coord = None
        self.profile_mean = None
        self.profile_median = None
        self.left_min = None
        self.right_min = None
        self.size_px = None
        self._current_cut_bbox = None
        self.left_ripples = []
        self.right_ripples = []

    def analyze(self):
        if self.gray is None:
            messagebox.showwarning("Analyze", "No image loaded")
            return

        # Find all coordinates with maximum intensity, average them (float coords)
        max_val = self.gray.max()
        ys, xs = np.where(self.gray == max_val)
        if ys.size == 0:
            messagebox.showwarning("Analyze", "No maximum found")
            return
        avg_x = xs.mean()
        avg_y = ys.mean()
        self.max_coord = (avg_x, avg_y)

        # Compute profile using current cut height from GUI
        try:
            cut_h = int(self.cut_spin.get())
            if cut_h < 1:
                cut_h = 1
        except Exception:
            cut_h = 10
            self.cut_spin.delete(0, "end")
            self.cut_spin.insert(0, "10")

        mode = self.mode_var.get()

        if mode == "classic":
            self._compute_profile_and_minima(avg_x, avg_y, cut_h)
            self._show_plots()
        else:
            self._analyze_young()
        info = f"Max at x={avg_x:.2f}, y={avg_y:.2f} (averaged). "
        if self.size_px is not None:
            info += f"Size ≈ {self.size_px} px (minima at {self.left_min} and {self.right_min})."
        else:
            info += "Could not determine both primary minima."
        self.info_var.set(info)

    def update_cut(self):
        if self.gray is None:
            messagebox.showwarning("Update Cut", "No image loaded")
            return
        if self.max_coord is None:
            messagebox.showwarning("Update Cut", "Run Analyze first to detect peak.")
            return
        try:
            cut_h = int(self.cut_spin.get())
            if cut_h < 1:
                cut_h = 1
        except Exception:
            cut_h = 10
            self.cut_spin.delete(0, "end")
            self.cut_spin.insert(0, "10")

        x0, y0 = self.max_coord
        self._compute_profile_and_minima(x0, y0, cut_h)
        self._show_plots()
        info = f"Cut updated: height={cut_h} px. "
        if self.size_px is not None:
            info += f"Size ≈ {self.size_px} px (minima at {self.left_min} and {self.right_min})."
        else:
            info += "Could not determine both primary minima."
        self.info_var.set(info)

    def _compute_profile_and_minima(self, x0, y0, cut_h):
        """
        Compute horizontal intensity profiles and detect primary diffraction minima.

        A vertical band of height `cut_h` centered at (x0, y0) is extracted, and
        mean and median intensity profiles are computed along the horizontal axis.

        Primary minima are identified by scanning outward from the central peak
        using a plateau-aware criterion:
        - Candidate values must lie within [0, 3]
        - Neighboring values outside the plateau must not be lower
        - At least one neighbor must be strictly higher (ensures a true minimum)

        Fallback strategies:
        - Nearest detected local minima (via scipy or simple comparison)
        - Global minimum on each side of the peak

        Results are stored in instance attributes:
            profile_mean, profile_median,
            left_min, right_min, size_px,
            left_ripples, right_ripples

        Parameters
        ----------
        x0 : float
            X-coordinate of the central peak.
        y0 : float
            Y-coordinate of the central peak.
        cut_h : int
            Height of the vertical averaging band (in pixels).
        """
        rows, cols = self.gray.shape
        # center cut: try to make total height = cut_h
        half_down = (cut_h - 1) // 2
        half_up = cut_h - half_down - 1
        y_min = max(0, int(round(y0)) - half_down)
        y_max = min(rows, int(round(y0)) + half_up + 1)  # exclusive

        cut_block = self.gray[y_min:y_max, :]
        if cut_block.size == 0:
            self.profile_mean = None
            self.profile_median = None
            self.left_min = None
            self.right_min = None
            self.size_px = None
            self.left_ripples = []
            self.right_ripples = []
            return

        profile_mean = cut_block.mean(axis=0)
        profile_median = np.median(cut_block, axis=0)
        self.profile_mean = profile_mean
        self.profile_median = profile_median

        peak_idx = int(round(x0))
        L = profile_mean.size

        # Detect local minima indices across the 1D profile (used for fallback)
        if _HAS_SCIPY:
            minima = argrelextrema(profile_mean, np.less, order=3)[0]
        else:
            minima_list = []
            for i in range(1, L - 1):
                if profile_mean[i] < profile_mean[i - 1] and profile_mean[i] < profile_mean[i + 1]:
                    minima_list.append(i)
            minima = np.array(minima_list, dtype=int)

        # If no local minima found by above, use global minima on segments as possible fallbacks
        if minima.size == 0:
            left_min_guess = int(np.argmin(profile_mean[:peak_idx])) if peak_idx > 0 else None
            right_min_guess = (int(np.argmin(profile_mean[peak_idx + 1:])) + peak_idx + 1) if peak_idx < L - 1 else None
            mins = [i for i in (left_min_guess, right_min_guess) if i is not None]
            minima = np.array(mins, dtype=int) if mins else np.array([], dtype=int)

        # Save all detected minima as ripples (for storage)
        left_minima_all = minima[minima < peak_idx] if minima.size > 0 else np.array([], dtype=int)
        right_minima_all = minima[minima > peak_idx] if minima.size > 0 else np.array([], dtype=int)
        self.left_ripples = left_minima_all.tolist()
        self.right_ripples = right_minima_all.tolist()

        # Candidate range and tolerances
        min_val_allowed = 0.0
        max_val_allowed = 3.0
        eps = 1e-8

        # Helper: detect plateau around index i (returns left_idx, right_idx inclusive)
        def plateau_bounds(i):
            val = profile_mean[i]
            # expand left
            Lb = i
            while Lb - 1 >= 0 and abs(profile_mean[Lb - 1] - val) <= eps:
                Lb -= 1
            # expand right
            Rb = i
            while Rb + 1 < L and abs(profile_mean[Rb + 1] - val) <= eps:
                Rb += 1
            return Lb, Rb, val

        # Left primary minima: scan from peak-1 downwards
        self.left_min = None
        if peak_idx > 0:
            for i in range(peak_idx - 1, -1, -1):
                val = profile_mean[i]
                if min_val_allowed <= val <= max_val_allowed:
                    Lb, Rb, v = plateau_bounds(i)
                    left_out = Lb - 1
                    right_out = Rb + 1
                    left_ok = (left_out < 0) or (profile_mean[left_out] >= v - eps)
                    right_ok = (right_out >= L) or (profile_mean[right_out] >= v - eps)
                    # require at least one surrounding value strictly greater to ensure a minimum/edge
                    left_strict = (left_out >= 0) and (profile_mean[left_out] > v + eps)
                    right_strict = (right_out < L) and (profile_mean[right_out] > v + eps)
                    if left_ok and right_ok and (left_strict or right_strict):
                        # choose plateau index closest to peak -> rightmost index Rb
                        self.left_min = int(min(Rb, peak_idx - 1))
                        break
            # fallback: nearest detected minima left of peak
            if self.left_min is None and left_minima_all.size > 0:
                self.left_min = int(left_minima_all[-1])
            # final fallback: global minimum on left segment
            if self.left_min is None and peak_idx > 0:
                self.left_min = int(np.argmin(profile_mean[:peak_idx]))

        # Right primary minima: scan from peak+1 upwards
        self.right_min = None
        if peak_idx < L - 1:
            for i in range(peak_idx + 1, L):
                val = profile_mean[i]
                if min_val_allowed <= val <= max_val_allowed:
                    Lb, Rb, v = plateau_bounds(i)
                    left_out = Lb - 1
                    right_out = Rb + 1
                    left_ok = (left_out < 0) or (profile_mean[left_out] >= v - eps)
                    right_ok = (right_out >= L) or (profile_mean[right_out] >= v - eps)
                    left_strict = (left_out >= 0) and (profile_mean[left_out] > v + eps)
                    right_strict = (right_out < L) and (profile_mean[right_out] > v + eps)
                    if left_ok and right_ok and (left_strict or right_strict):
                        # choose plateau index closest to peak -> leftmost index Lb
                        self.right_min = int(max(Lb, peak_idx + 1))
                        break
            # fallback: nearest detected minima right of peak
            if self.right_min is None and right_minima_all.size > 0:
                self.right_min = int(right_minima_all[0])
            # final fallback: global minimum on right segment
            if self.right_min is None and peak_idx < L - 1:
                seg = profile_mean[peak_idx + 1:]
                if seg.size > 0:
                    self.right_min = int(np.argmin(seg)) + peak_idx + 1

        if self.left_min is not None and self.right_min is not None:
            self.size_px = self.right_min - self.left_min
        else:
            self.size_px = None

        # Save cut rectangle coords for plotting (x, y, width, height)
        self._current_cut_bbox = (0, y_min, cols, y_max - y_min)
        
        
    def _analyze_young(self):
        """
        Specialized analysis for Young double-slit diffraction patterns.

        Steps:
        - Detect fringe orientation using image gradients
        - Rotate image to align fringes vertically
        - Compute vertical intensity profile
        - Extract:
            1. Interfringe spacing (distance between bright fringes)
            2. Airy envelope size (central diffraction envelope)
        """
        

        img = self.gray.copy()

        # --- 1. Compute gradients ---
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        angles = np.arctan2(gy, gx)
        dominant_angle = np.median(angles)

        # Fringe direction is perpendicular to gradient
        theta = dominant_angle  # instead of + π/2
        angle_deg = np.degrees(theta)
        # --- 2. Rotate image ---
        h, w = img.shape
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

        # --- 3. Compute vertical profile ---
        vertical_profile = np.mean(rotated, axis=1)
        horizontal_profile = np.mean(rotated, axis=0)

        center_y = np.argmax(savgol_filter(vertical_profile, 201, 3))
        center_x = np.argmax(savgol_filter(horizontal_profile, 201, 3))

        self.max_coord = (center_x, center_y)
        profile=(center_x, center_y)

        # Smooth
        profile_smooth = savgol_filter(profile, 51, 3)

        # --- 4. Find fringes (peaks) ---
        peaks, _ = find_peaks(profile_smooth, distance=10)

        if len(peaks) > 1:
            interfringes = np.diff(peaks)
            interfringe = np.mean(interfringes)
        else:
            interfringe = None

        # --- 5. Airy envelope detection ---
        # Strong smoothing to remove fringes
        envelope = savgol_filter(profile, 201, 3)

        # Find minima around center
        center_idx = np.argmax(envelope)

        minima = argrelextrema(envelope, np.less)[0]

        left = minima[minima < center_idx]
        right = minima[minima > center_idx]

        airy_size = None
        if len(left) > 0 and len(right) > 0:
            left_min = left[-1]
            right_min = right[0]
            airy_size = right_min - left_min

        # --- Store results ---
        self.young_interfringe = interfringe
        self.young_airy_size = airy_size
        self.young_profile = profile_smooth
        self.young_peaks = peaks

        # --- Visualization (enhanced, classical-like) ---

        plt.ion()

        # === 1. Rotated image with overlays ===
        plt.figure("Young - Rotated Image Analysis")
        plt.clf()

        plt.imshow(rotated, cmap="gray", origin="upper")

        h, w = rotated.shape

        # Central horizontal axis (used for profile averaging)
        y_center = h // 2
        plt.axhline(y_center, color='yellow', linestyle='--', label="Profile axis")

        # Show averaging band
        band_half = 5
        plt.fill_between(
            np.arange(w),
            y_center - band_half,
            y_center + band_half,
            color='yellow',
            alpha=0.2,
            label="Averaging band"
        )

        # Plot detected fringes
        if peaks is not None:
            plt.scatter(peaks, [y_center]*len(peaks),
                        color='red', s=20, label="Fringes")

        # Plot Airy boundaries
        if airy_size is not None:
            center_idx = np.argmax(envelope)
            left_min = left[-1] if len(left) > 0 else None
            right_min = right[0] if len(right) > 0 else None

            if left_min is not None:
                plt.axvline(left_min, color='green', linestyle=':', label="Airy boundary")
            if right_min is not None:
                plt.axvline(right_min, color='green', linestyle=':')

        plt.title(f"Rotated analysis (angle ≈ {angle_deg:.1f}°)")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.legend(loc="upper right")


        # === 2. High contrast rotated image ===
        plt.figure("Young - High Contrast")
        plt.clf()

        p_low, p_high = np.percentile(rotated, (1, 99))
        norm = np.clip((rotated - p_low) / (p_high - p_low), 0, 1)

        log_scaled = np.log1p(norm * 200) / np.log1p(200)
        stretched = np.power(log_scaled, 0.35)

        plt.imshow(stretched, cmap="gray", origin="upper")

        plt.axhline(y_center, color='cyan', linestyle='--')

        plt.title("High-contrast (Young)")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")


        # === 3. Profile (your existing but improved) ===
        plt.figure("Young - Profile")
        plt.clf()

        x = np.arange(len(profile_smooth))
        plt.plot(x, profile_smooth, label="Profile (smoothed)", lw=1.5)

        # Peaks
        if peaks is not None:
            plt.scatter(peaks, profile_smooth[peaks],
                        color='red', label="Fringes")

        # Airy envelope
        plt.plot(x, envelope, linestyle='--', label="Envelope")

        # Airy boundaries
        if airy_size is not None:
            plt.axvline(left_min, color='green', linestyle=':')
            plt.axvline(right_min, color='green', linestyle=':')
            plt.text((left_min + right_min)/2,
                     np.max(profile_smooth)*0.8,
                     f"Airy ≈ {airy_size}px",
                     ha='center',
                     bbox=dict(facecolor='white', alpha=0.7))

        # Interfringe
        if interfringe is not None:
            plt.title(f"Interfringe ≈ {interfringe:.2f} px")

        plt.xlabel("x (pixels)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid()

        plt.show()

        # --- Update GUI ---
        info = "Young mode: "
        if interfringe is not None:
            info += f"Interfringe ≈ {interfringe:.2f} px, "
        if airy_size is not None:
            info += f"Airy size ≈ {airy_size} px"
        self.info_var.set(info)



    
    
    def _show_plots(self):
        """
        Display analysis results using matplotlib (non-blocking).

        Generates three figures:
        1. Original grayscale image with:
            - Detected maximum (red marker)
            - Averaging cut region (rectangle)
        2. High-contrast version of the image using a nonlinear stretch
           (log/asinh-like transformation) to enhance low-intensity details
        3. Horizontal intensity profile showing:
            - Mean and median curves
            - Peak position
            - Primary left and right minima
            - Estimated diffraction size (if available)

        Peak coordinates are shown in figure titles to avoid cluttering plots.
        """
        if self.gray is None or self.profile_mean is None:
            return

        # Image window (original)
        plt.figure("Diffraction Image")
        plt.clf()
        plt.imshow(self.gray, cmap="gray", origin="upper")
        if self.max_coord is not None:
            mx, my = self.max_coord
            plt.plot(mx, my, "rx", markersize=8, label="Max (averaged)")
        if self._current_cut_bbox is not None:
            x, y, w, h = self._current_cut_bbox
            rect = Rectangle((x, y), w, h, facecolor="none", edgecolor='yellow', linewidth=1.5, linestyle='--', alpha=0.9)
            plt.gca().add_patch(rect)
        # put coordinates in the title so they don't impair visualization
        if self.max_coord is not None:
            plt.title(f"Image — Max at x={self.max_coord[0]:.1f}, y={self.max_coord[1]:.1f}")
        else:
            plt.title("Image")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.legend(loc="upper right")

        # High-contrast image window (exaggerated stretch)
        plt.figure("Diffraction Image (High Contrast - Exaggerated)")
        plt.clf()
        # robust percentiles for baseline stretch
        p_low, p_high = np.percentile(self.gray, (1.0, 99.0))
        if p_high <= p_low:
            p_low, p_high = self.gray.min(), self.gray.max()
        # normalize to [0,1]
        norm = (self.gray - p_low) / (p_high - p_low)
        norm = np.clip(norm, 0.0, 1.0)

        # Exaggerated stretch:
        scale = 200.0
        log_scaled = np.log1p(norm * scale) / np.log1p(scale)
        gamma = 0.35
        stretched = np.power(log_scaled, gamma)
        stretched = np.clip(stretched, 0.0, 1.0)

        plt.imshow(stretched, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
        if self.max_coord is not None:
            mx, my = self.max_coord
            plt.plot(mx, my, "rx", markersize=8)
        if self._current_cut_bbox is not None:
            x, y, w, h = self._current_cut_bbox
            rect = Rectangle((x, y), w, h, facecolor="none", edgecolor='cyan', linewidth=1.0, linestyle='-', alpha=0.9)
            plt.gca().add_patch(rect)
        if self.max_coord is not None:
            plt.title(f"High-contrast view — Max at x={self.max_coord[0]:.1f}, y={self.max_coord[1]:.1f}")
        else:
            plt.title("High-contrast view")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")

        # Profile window
        plt.figure("Horizontal Profile")
        plt.clf()
        x = np.arange(self.profile_mean.size)
        plt.plot(x, self.profile_mean, color="blue", lw=1.5, label="Mean (cut)")
        plt.plot(x, self.profile_median, color="magenta", lw=1.0, linestyle="--", label="Median (cut)")
        if self.max_coord is not None:
            peak_x = int(round(self.max_coord[0]))
            plt.axvline(peak_x, color="red", linestyle="--", label="Peak x")

        # Plot only the primary minima (first left / first right)
        if self.left_min is not None:
            plt.scatter([self.left_min], [self.profile_mean[self.left_min]], color="green", zorder=6, label="Left primary minima")
            plt.axvline(self.left_min, color="green", linestyle=":", lw=1.0)
        if self.right_min is not None:
            plt.scatter([self.right_min], [self.profile_mean[self.right_min]], color="orange", zorder=6, label="Right primary minima")
            plt.axvline(self.right_min, color="orange", linestyle=":", lw=1.0)

        if self.size_px is not None and self.left_min is not None and self.right_min is not None:
            mid = (self.left_min + self.right_min) / 2
            ymax = self.profile_mean.max()
            plt.text(mid, ymax * 0.8, f"Size: {self.size_px} px", ha="center", bbox=dict(facecolor="white", alpha=0.8))

        # include peak coordinates in the profile title (no text overlays)
        if self.max_coord is not None:
            plt.title(f"Horizontal intensity profile — Peak x={self.max_coord[0]:.1f}, y={self.max_coord[1]:.1f}")
        else:
            plt.title("Horizontal intensity profile")
        plt.xlabel("x (pixels)")
        plt.ylabel("Intensity")
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.ion()
        plt.show()

    def save_size(self):
        """
        Store the current measurement in an internal history log displayed in the GUI.

        The saved entry depends on the current mode:
        - Classic mode: diffraction size between primary minima
        - Young mode: interfringe and Airy envelope size

        Entries are appended to a text panel for easy tracking.
        """
        mode = self.mode_var.get()

        if mode == "classic":
            if self.size_px is None:
                messagebox.showwarning("Save", "No measurement available.")
                return

            entry = (
                f"[Classic] Size = {self.size_px} px "
                f"(minima: {self.left_min}, {self.right_min})"
            )

        else:  # Young mode
            if getattr(self, "young_interfringe", None) is None and getattr(self, "young_airy_size", None) is None:
                messagebox.showwarning("Save", "No measurement available.")
                return

            entry = "[Young] "

            if self.young_interfringe is not None:
                entry += f"Interfringe = {self.young_interfringe:.2f} px "

            if self.young_airy_size is not None:
                entry += f"| Airy size = {self.young_airy_size} px"

        # Add image name if available
        if self.image_path:
            entry = f"{os.path.basename(self.image_path)} → " + entry

        # Store in history
        self.history.append(entry)

        # Display in text box
        self.history_text.insert(tk.END, entry + "\n")
        self.history_text.see(tk.END)

if __name__ == "__main__":
    app = DiffractionViewer()
    app.mainloop()