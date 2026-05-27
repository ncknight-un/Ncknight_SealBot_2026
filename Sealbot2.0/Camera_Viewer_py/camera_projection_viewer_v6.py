"""
Camera Projection Cone Viewer  v6
===================================
Interactive GUI for visualizing horizontal camera projection cones,
depth-of-field, and AprilTag detection ranges.

Units: millimeters (mm)

Changes in v6:
    - Dropdown lens selection now syncs the focal length textbox
    - Removed near/far plane lines/bands from the cone plot
      (z_far still caps the view at 300 mm by default)
    - Added Tag Angle input (degrees): effective projected size scales
      with |cos(angle)| — models a tag rotated around its vertical axis
    - Fixed tag size not initialising correctly at startup
"""

import json
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons, TextBox
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*')

# ─────────────────────────────────────────────────────────────────
# Lens catalogue
# ─────────────────────────────────────────────────────────────────
LENS_TYPES = {
    "2.0 mm f/1.2":  dict(f_mm=2.0,  f_number=1.2),
    "2.0 mm f/1.4":  dict(f_mm=2.0,  f_number=1.4),
    "2.0 mm f/1.8":  dict(f_mm=2.0,  f_number=1.8),
    "5 mm f/1.2":    dict(f_mm=5.0,  f_number=1.2),
    "6 mm f/1.8":    dict(f_mm=6.0,  f_number=1.8),
    "6 mm f/4":      dict(f_mm=6.0,  f_number=4.0),
    "6 mm f/5.6":      dict(f_mm=6.0,  f_number=5.6),
    "6 mm f/8":      dict(f_mm=6.0,  f_number=8.0),
    "6 mm f/16":     dict(f_mm=6.0,  f_number=16.0),
    "8 mm f/1.4":    dict(f_mm=8.0,  f_number=1.4),
    "8 mm f/1.8":    dict(f_mm=8.0,  f_number=1.8),
    "8.0 mm f/5.6":  dict(f_mm=8.0,  f_number=5.6),
    "8.0 mm f/8.0":  dict(f_mm=8.0,  f_number=8.0),
    "8.5 mm f/4.0":  dict(f_mm=8.5,  f_number=4.0),
    "8.5 mm f/5.6":  dict(f_mm=8.5,  f_number=5.6),
    "8.5 mm f/8.0":  dict(f_mm=8.5,  f_number=8.0),
    "8.5 mm f/12.0": dict(f_mm=8.5,  f_number=12.0),
    "12 mm f/4":     dict(f_mm=12.0, f_number=4.0),
    "12 mm f/5.6":   dict(f_mm=12.0, f_number=5.6),
    "12 mm f/8":     dict(f_mm=12.0, f_number=8.0),
    "12 mm f/11":    dict(f_mm=12.0, f_number=11.0),
    "12 mm f/16":    dict(f_mm=12.0, f_number=16.0),
    "16 mm f/4":     dict(f_mm=16.0, f_number=4.0),
    "16 mm f/8":     dict(f_mm=16.0, f_number=8.0),
    "16 mm f/11":    dict(f_mm=16.0, f_number=11.0),
    "25 mm f/2.8":   dict(f_mm=25.0, f_number=2.8),
    "25 mm f/8":     dict(f_mm=25.0, f_number=8.0),
    "50 mm f/4":     dict(f_mm=50.0, f_number=4.0),
}

# ─────────────────────────────────────────────────────────────────
# AprilTag families and sizes
# ─────────────────────────────────────────────────────────────────
APRILTAG_FAMILIES = {
    "Tag36h11": [
        ("5 mm",   5.0),  ("10 mm",  10.0), ("20 mm",  20.0), ("40 mm",  40.0),
        ("60 mm",  60.0), ("80 mm",  80.0), ("120 mm", 120.0), ("160 mm", 160.0),
        ("200 mm", 200.0),("250 mm", 250.0),("300 mm", 300.0),
    ],
    "Tag25h9": [
        ("5 mm",   5.0),  ("10 mm",  10.0), ("15 mm",  15.0), ("30 mm",  30.0),
        ("60 mm",  60.0), ("90 mm",  90.0), ("120 mm", 120.0), ("180 mm", 180.0),
        ("240 mm", 240.0),("300 mm", 300.0),
    ],
    "Tag16h5": [
        ("5 mm",   5.0),  ("10 mm",  10.0), ("20 mm",  20.0), ("40 mm",  40.0),
        ("60 mm",  60.0), ("100 mm", 100.0),("150 mm", 150.0),("200 mm", 200.0),
    ],
    "TagStd41h12": [
        ("10 mm",  10.0), ("20 mm",  20.0), ("30 mm",  30.0), ("60 mm",  60.0),
        ("100 mm", 100.0),("150 mm", 150.0),("200 mm", 200.0),("300 mm", 300.0),
    ],
    "Circle21h7": [
        ("10 mm",  10.0), ("20 mm",  20.0), ("50 mm",  50.0), ("80 mm",  80.0),
        ("100 mm", 100.0),("150 mm", 150.0),("200 mm", 200.0),
    ],
}

DETECTION_THRESHOLDS = [
    (10, '#e67e22', 'Poor (10 px)'),
    (20, '#f1c40f', 'Fair (20 px)'),
    (40, '#2ecc71', 'Good (40 px)'),
    (80, '#3498db', 'Excellent (80 px)'),
]

TAG_COLORS = ['#8e44ad', '#c0392b', '#16a085', '#d35400', '#2980b9',
              '#27ae60', '#8e44ad', '#f39c12']


# ─────────────────────────────────────────────────────────────────
# Angle helper
# ─────────────────────────────────────────────────────────────────

def effective_tag_size(tag_size_mm, angle_deg):
    """
    Effective width visible to the camera when the tag is yawed by angle_deg.
    At 0° the full tag face is seen; at 90° the tag is edge-on (nothing seen).
    Uses |cos(angle)| to model the projected width.
    """
    return tag_size_mm * abs(np.cos(np.radians(float(angle_deg))))


# ─────────────────────────────────────────────────────────────────
# Lightweight pure-matplotlib dropdown
# ─────────────────────────────────────────────────────────────────

class DropdownMenu:
    _registry: list = []

    def __init__(self, fig, btn_rect, panel_rect, options,
                 initial=None, on_select=None,
                 btn_color='#ecf0f1', panel_facecolor='#f8f9fa',
                 active_color='#2c3e50', label_fontsize=7.5,
                 header_color='#2c3e50'):
        self.fig       = fig
        self.options   = list(options)
        self.on_select = on_select
        self._open     = False
        self._current  = initial if initial in self.options else self.options[0]

        self._ax_btn = fig.add_axes(btn_rect)
        self._ax_btn.set_zorder(10)
        self._btn = Button(self._ax_btn, self._trunc(self._current),
                           color=btn_color, hovercolor='#d5d8dc')
        self._btn.label.set_fontsize(label_fontsize)
        self._btn.label.set_color(header_color)
        self._btn.on_clicked(self._toggle)

        self._ax_panel = fig.add_axes(panel_rect)
        self._ax_panel.set_zorder(20)
        self._ax_panel.set_visible(False)

        active_idx = (self.options.index(self._current)
                      if self._current in self.options else 0)
        self._radio = RadioButtons(self._ax_panel, self.options,
                                   active=active_idx, activecolor=active_color)
        for lbl in self._radio.labels:
            lbl.set_fontsize(label_fontsize)
        self._radio.on_clicked(self._on_select)
        DropdownMenu._registry.append(self)

    @staticmethod
    def _trunc(text, n=22):
        return text if len(text) <= n else text[:n-1] + '…'

    def _toggle(self, _=None):
        if self._open:
            self.close()
        else:
            for dd in DropdownMenu._registry:
                if dd is not self and dd._open:
                    dd.close()
            self._open = True
            self._ax_panel.set_visible(True)
            self.fig.canvas.draw_idle()

    def _on_select(self, label):
        self._current = label
        self._btn.label.set_text(self._trunc(label))
        self.close()
        if self.on_select:
            self.on_select(label)

    def close(self):
        self._open = False
        self._ax_panel.set_visible(False)
        self.fig.canvas.draw_idle()

    def set_options(self, options, new_selection=None):
        self.options  = list(options)
        sel           = new_selection if new_selection in self.options else self.options[0]
        self._current = sel
        self._btn.label.set_text(self._trunc(sel))
        self._ax_panel.clear()
        active_idx = self.options.index(sel)
        self._radio = RadioButtons(self._ax_panel, self.options,
                                   active=active_idx,
                                   activecolor=self._radio.activecolor)
        for lbl in self._radio.labels:
            lbl.set_fontsize(7.5)
        self._radio.on_clicked(self._on_select)
        self.close()

    @property
    def value(self):
        return self._current


# ─────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────

def compute_fov(fx, fy, iw, ih):
    hfov = np.degrees(2 * np.arctan(iw / (2 * fx)))
    vfov = np.degrees(2 * np.arctan(ih / (2 * fy)))
    diag = np.degrees(2 * np.arctan(
        np.sqrt(iw**2 + ih**2) / (2 * np.sqrt(fx * fy))))
    return hfov, vfov, diag


def frustum_half_width(fx, iw, z_mm):
    return z_mm * iw / (2 * fx)


def get_K_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def detection_distance(f_px, tag_mm, min_pixels):
    if min_pixels <= 0 or tag_mm <= 0:
        return float('inf')
    return f_px * tag_mm / min_pixels


def compute_intrinsics(f_mm, sensor_w, sensor_h, iw, ih):
    fx = (f_mm / sensor_w) * iw
    fy = (f_mm / sensor_h) * ih
    return fx, fy, iw / 2.0, ih / 2.0


def pixel_size_mm(sensor_w_mm, iw_px):
    return sensor_w_mm / iw_px if iw_px > 0 else 0.004


def auto_coc(sensor_w_mm, sensor_h_mm):
    return np.sqrt(sensor_w_mm**2 + sensor_h_mm**2) / 1500.0


def compute_dof(fx_mm_equiv, f_number, focus_dist_mm, coc_mm):
    f, N, c = fx_mm_equiv, f_number, coc_mm
    if c <= 0 or f <= 0 or focus_dist_mm <= 0:
        return 0.0, float('inf'), float('inf')
    H    = (f * f) / (N * c) + f
    s    = focus_dist_mm
    near = (s * (H - f)) / (H + s - 2*f) if (H + s - 2*f) > 0 else s / 2
    far  = (s * (H - f)) / (H - f - s)   if (H - f - s) > 0    else float('inf')
    if s >= H:
        far = float('inf')
    return max(1.0, near), far, H


def parse_float_list(text):
    out = []
    for tok in str(text).split(','):
        tok = tok.strip()
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


# ─────────────────────────────────────────────────────────────────
# Artist cache
# ─────────────────────────────────────────────────────────────────

class ArtistCache:
    def __init__(self):
        self._groups: dict = {}

    def add(self, group, artist):
        self._groups.setdefault(group, []).append(artist)

    def clear(self, *groups):
        targets = groups if groups else list(self._groups.keys())
        for g in targets:
            for a in self._groups.pop(g, []):
                try:
                    a.remove()
                except Exception:
                    pass

    def clear_all(self):
        self.clear(*list(self._groups.keys()))


# ─────────────────────────────────────────────────────────────────
# Static builders
# ─────────────────────────────────────────────────────────────────

def build_cone_static(ax, cache, fx, iw, z_near_mm, z_far_mm,
                      tag_size_mm, eff_tag_size_mm):
    """
    Frustum fill + boundary + optical axis + detection zones + FOV arc.
    Near/far plane visual bands are NOT drawn (v6 change).
    Detection distances use eff_tag_size_mm (angle-corrected).
    """
    cache.clear('cone_static')

    hfov_rad = 2 * np.arctan(iw / (2 * fx))
    z_vals   = np.linspace(0, z_far_mm, 800)
    half_w   = z_vals * np.tan(hfov_rad / 2)
    hf       = z_far_mm * np.tan(hfov_rad / 2)

    def add(a):
        cache.add('cone_static', a)
        return a

    add(ax.fill_between(z_vals, -half_w, half_w,
                        alpha=0.10, color='steelblue', zorder=1))
    add(ax.plot(z_vals,  half_w, color='steelblue', lw=2.0,
                label='Frustum boundary')[0])
    add(ax.plot(z_vals, -half_w, color='steelblue', lw=2.0)[0])
    add(ax.axhline(0, color='gray', lw=0.8, ls=':', alpha=0.4, zorder=1))

    zone_patches = []
    if eff_tag_size_mm > 0:
        prev_z = z_near_mm
        for min_px, color, lbl in DETECTION_THRESHOLDS:
            z_t = min(detection_distance(fx, eff_tag_size_mm, min_px), z_far_mm)
            if z_t > prev_z:
                add(ax.axvspan(prev_z, z_t, alpha=0.06, color=color, zorder=1))
                add(ax.axvline(z_t, color=color, lw=1.2, ls='--',
                               alpha=0.80, zorder=2))
                add(ax.text(z_t, hf * 0.88,
                            f'{min_px}px\n{z_t:.0f}',
                            fontsize=6.5, color=color, ha='center', va='top',
                            bbox=dict(fc='white', ec='none', alpha=0.5, pad=1)))
            prev_z = z_t
            zone_patches.append(
                mpatches.Patch(color=color, alpha=0.5, label=lbl))
            if z_t >= z_far_mm:
                break

    arc_r = z_far_mm * 0.13
    theta  = np.linspace(-hfov_rad / 2, hfov_rad / 2, 120)
    add(ax.plot([0, arc_r * np.cos(hfov_rad/2)],
                [0,  arc_r * np.sin(hfov_rad/2)],
                color='#f39c12', lw=1.3, alpha=0.8)[0])
    add(ax.plot([0, arc_r * np.cos(hfov_rad/2)],
                [0, -arc_r * np.sin(hfov_rad/2)],
                color='#f39c12', lw=1.3, alpha=0.8)[0])
    add(ax.plot(arc_r * np.cos(theta), arc_r * np.sin(theta),
                color='#f39c12', lw=2.0)[0])
    add(ax.annotate(f'H-FOV\n{np.degrees(hfov_rad):.1f}°',
                    xy=(arc_r * 1.12, 0), fontsize=8,
                    color='#f39c12', va='center', ha='left'))
    add(ax.scatter([0], [0], s=80, color='steelblue', zorder=6,
                   label='Camera origin'))

    ax.set_xlabel('Depth Z (mm)', fontsize=9)
    ax.set_ylabel('Horizontal extent (mm)', fontsize=9)
    ax.set_title('Top-down view — Horizontal FOV cone',
                 fontsize=10, fontweight='bold', pad=4)
    ax.set_xlim(0, z_far_mm * 1.10)
    ax.grid(True, alpha=0.18)

    return zone_patches


def build_tag_static(ax, cache, fx, tag_size_mm, eff_tag_size_mm,
                     z_far_mm, family, tag_label, angle_deg):
    """Detection-range curve using angle-corrected effective size."""
    cache.clear('tag_static')

    def add(a):
        cache.add('tag_static', a)
        return a

    if eff_tag_size_mm <= 0:
        add(ax.text(0.5, 0.5, 'Tag angle ≥ 90° — no projection visible',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray'))
        ax.set_title('AprilTag detection range', fontsize=10,
                     fontweight='bold', pad=4)
        return

    z_max_plot = min(detection_distance(fx, eff_tag_size_mm, 4),
                     z_far_mm * 1.5, 8000)
    z_vals = np.linspace(1, z_max_plot, 800)
    tag_px = fx * eff_tag_size_mm / z_vals

    sorted_t = sorted(DETECTION_THRESHOLDS, key=lambda x: x[0])
    for i, (min_px, color, _lbl) in enumerate(sorted_t):
        next_px = sorted_t[i+1][0] if i+1 < len(sorted_t) else tag_px.max()*2
        mask = (tag_px >= min_px) & (tag_px < next_px)
        if mask.any():
            add(ax.fill_between(z_vals, min_px,
                                np.where(mask, tag_px, min_px),
                                alpha=0.18, color=color))

    angle_str = (f'  (angle {angle_deg:.0f}°, eff {eff_tag_size_mm:.1f} mm)'
                 if abs(angle_deg) > 0.5 else '')
    add(ax.plot(z_vals, tag_px, color='#2c3e50', lw=2.2,
                label=f'{family}  {tag_label}{angle_str}')[0])

    for min_px, color, lbl in DETECTION_THRESHOLDS:
        z_t = detection_distance(fx, eff_tag_size_mm, min_px)
        add(ax.axhline(min_px, color=color, lw=1.2, ls='--',
                       alpha=0.85, label=lbl))
        if z_t <= z_max_plot:
            add(ax.axvline(z_t, color=color, lw=0.8, ls=':', alpha=0.55))
            add(ax.text(z_t, min_px + 1.2, f'{z_t:.0f}',
                        fontsize=6.5, color=color, ha='center', va='bottom'))

    ax.set_xlabel('Distance Z (mm)', fontsize=9)
    ax.set_ylabel('Tag projected size (px)', fontsize=9)
    angle_title = (f' — {angle_deg:.0f}° angle' if abs(angle_deg) > 0.5 else '')
    ax.set_title(f'Detection range — {family}  {tag_label}{angle_title}',
                 fontsize=10, fontweight='bold', pad=4)
    ax.set_ylim(0, min(tag_px.max() * 1.15, 500))
    ax.set_xlim(0, z_max_plot)
    ax.grid(True, alpha=0.18)


# ─────────────────────────────────────────────────────────────────
# Dynamic overlays
# ─────────────────────────────────────────────────────────────────

def update_cone_dynamic(ax, cache, fx, iw, z_far_mm,
                        dof_near, dof_far, tags,
                        eff_tag_size_mm, zone_patches):
    cache.clear('cone_dynamic')
    hfov_rad = 2 * np.arctan(iw / (2 * fx))
    hf       = z_far_mm * np.tan(hfov_rad / 2)

    def add(a):
        cache.add('cone_dynamic', a)
        return a

    if dof_near is not None and dof_far is not None:
        dn = float(np.clip(dof_near, 0, z_far_mm * 1.05))
        df = (float(np.clip(dof_far, 0, z_far_mm * 1.05))
              if dof_far != float('inf') else z_far_mm * 1.05)
        z_dof  = np.linspace(dn, df, 400)
        hw_dof = z_dof * np.tan(hfov_rad / 2)
        far_lbl = '∞' if dof_far == float('inf') else f'{dof_far:.0f}'
        add(ax.fill_between(z_dof, -hw_dof, hw_dof,
                            alpha=0.22, color='#00bcd4', zorder=2,
                            label=f'DoF  {dn:.0f}–{far_lbl} mm'))
        add(ax.axvline(dn, color='#00bcd4', lw=2.0, ls='-.', alpha=0.9,
                       zorder=3, label=f'DoF near  {dn:.0f} mm'))
        if dof_far != float('inf') and df <= z_far_mm * 1.05:
            add(ax.axvline(df, color='#0097a7', lw=2.0, ls='-.', alpha=0.9,
                           zorder=3, label=f'DoF far  {df:.0f} mm'))
        else:
            add(ax.axvline(z_far_mm * 1.04, color='#0097a7', lw=1.5, ls=':',
                           alpha=0.5, zorder=3, label='DoF far → ∞'))

    for i, (tz, tlat) in enumerate(tags):
        if tz <= 0:
            continue
        color   = TAG_COLORS[i % len(TAG_COLORS)]
        hw_at_z = tz * np.tan(hfov_rad / 2)
        in_fov  = abs(tlat) <= hw_at_z
        mcol    = color if in_fov else '#c0392b'
        px_size = fx * eff_tag_size_mm / tz if eff_tag_size_mm > 0 else None

        add(ax.scatter([tz], [tlat], s=160, marker='*', color=mcol, zorder=8,
                       label=f'Tag {i+1} @ ({tz:.0f}, {tlat:.0f}) mm'))
        fov_str = 'in FOV' if in_fov else 'OUT of FOV'
        px_str  = f'{px_size:.1f} px' if px_size else ''
        add(ax.annotate(f'T{i+1}: {fov_str}\n{px_str}',
                        xy=(tz, tlat),
                        xytext=(tz + z_far_mm * 0.04,
                                tlat + hw_at_z * 0.15),
                        fontsize=8, color=mcol, va='bottom', ha='left',
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=mcol, lw=1.2),
                        bbox=dict(fc='white', ec=mcol, alpha=0.85,
                                  pad=2, boxstyle='round')))

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + zone_patches,
              loc='upper left', fontsize=7, framealpha=0.70,
              ncol=2, columnspacing=0.8, handlelength=1.5)


def update_tag_dynamic(ax, cache, fx, eff_tag_size_mm, z_far_mm,
                       tags, dof_near, dof_far):
    cache.clear('tag_dynamic')
    if eff_tag_size_mm <= 0:
        return

    z_max_plot = min(detection_distance(fx, eff_tag_size_mm, 4),
                     z_far_mm * 1.5, 8000)

    def add(a):
        cache.add('tag_dynamic', a)
        return a

    if dof_near is not None and dof_far is not None:
        dn  = float(np.clip(dof_near, 0, z_max_plot))
        df_ = (float(np.clip(dof_far, 0, z_max_plot))
               if dof_far != float('inf') else z_max_plot)
        add(ax.axvspan(dn, df_, alpha=0.12, color='#00bcd4', label='DoF range'))

    for i, (tz, _) in enumerate(tags):
        if tz <= 0 or tz > z_max_plot:
            continue
        color   = TAG_COLORS[i % len(TAG_COLORS)]
        px_size = fx * eff_tag_size_mm / tz
        add(ax.scatter([tz], [px_size], s=120, marker='*',
                       color=color, zorder=7,
                       label=f'Tag {i+1}  {px_size:.1f} px'))
        add(ax.annotate(f'T{i+1}: {px_size:.1f} px',
                        xy=(tz, px_size),
                        xytext=(tz * 1.06, px_size * 1.12),
                        fontsize=8, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.1)))

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              framealpha=0.65, ncol=2,
              columnspacing=0.8, handlelength=1.5)


# ─────────────────────────────────────────────────────────────────
# Metrics panel
# ─────────────────────────────────────────────────────────────────

def draw_metrics(ax, fx, fy, cx, cy, iw, ih,
                 z_near, z_far, tag_size, eff_tag_size, angle_deg,
                 lens_name, family, tag_label,
                 dof_near=None, dof_far=None, hyperfocal=None,
                 focus_dist=None, tags=None):
    ax.clear()
    ax.axis('off')

    hfov, vfov, diag = compute_fov(fx, fy, iw, ih)
    hw_n = frustum_half_width(fx, iw, z_near)
    hw_f = frustum_half_width(fx, iw, z_far)
    li   = LENS_TYPES[lens_name]

    rows = [
        ('hdr', '── Lens ──────────────────────'),
        ('kv',  'Focal length', f"{li['f_mm']:.1f} mm"),
        ('kv',  'f-number',     f"f/{li['f_number']}"),
        ('gap', ''),
        ('hdr', '── Intrinsics ────────────────'),
        ('kv',  'fx / fy',    f"{fx:.0f} / {fy:.0f} px"),
        ('kv',  'cx / cy',    f"{cx:.0f} / {cy:.0f} px"),
        ('kv',  'Resolution', f"{int(iw)} × {int(ih)} px"),
        ('gap', ''),
        ('hdr', '── Field of View ─────────────'),
        ('kv',  'H-FOV',   f"{hfov:.1f}°"),
        ('kv',  'V-FOV',   f"{vfov:.1f}°"),
        ('kv',  'Diag FOV',f"{diag:.1f}°"),
        ('gap', ''),
        ('hdr', '── Frustum width ─────────────'),
        ('kv',  f'@ {z_near:.0f} mm', f"{2*hw_n:.1f} mm"),
        ('kv',  f'@ {z_far:.0f} mm',  f"{2*hw_f:.1f} mm"),
    ]

    if dof_near is not None:
        far_str = '∞' if dof_far == float('inf') else f'{dof_far:.0f} mm'
        hyp_str = '∞' if hyperfocal == float('inf') else f'{hyperfocal:.0f} mm'
        rows += [
            ('gap', ''),
            ('hdr', '── Depth of Field ────────────'),
            ('kv',  'Focus dist', f"{focus_dist:.0f} mm" if focus_dist else '—'),
            ('kvc', 'DoF near',   f"{dof_near:.0f} mm", '#00bcd4'),
            ('kvc', 'DoF far',    far_str,              '#0097a7'),
            ('kv',  'Hyperfocal', hyp_str),
            ('kv',  'DoF span',   ('∞' if dof_far == float('inf')
                                   else f'{dof_far - dof_near:.0f} mm')),
        ]

    if tag_size > 0:
        rows += [
            ('gap', ''),
            ('hdr', f'── {family}  {tag_label} ──'),
        ]
        if abs(angle_deg) > 0.5:
            rows.append(('kvc', 'Tag angle',
                          f'{angle_deg:.1f}° → eff {eff_tag_size:.1f} mm',
                          '#9b59b6'))
        for min_px, color, lbl in DETECTION_THRESHOLDS:
            z_t = detection_distance(fx, eff_tag_size, min_px)
            rows.append(('kvc', lbl, f"{z_t:.0f} mm", color))

    if tags:
        rows += [('gap', ''), ('hdr', '── Tag positions ─────────────')]
        for i, (tz, tlat) in enumerate(tags):
            if tz > 0 and eff_tag_size > 0:
                px = fx * eff_tag_size / tz
                rows.append(('kvc', f'Tag {i+1} ({tz:.0f},{tlat:.0f}) mm',
                              f'{px:.1f} px', TAG_COLORS[i % len(TAG_COLORS)]))

    rows += [
        ('gap', ''),
        ('hdr', '── K matrix ──────────────────'),
        ('mono', f" [[{fx:.0f}, 0, {cx:.0f}]"),
        ('mono', f"  [0, {fy:.0f}, {cy:.0f}]"),
        ('mono', f"  [0,  0,  1]]"),
    ]

    y, dy = 0.98, 0.040
    for row in rows:
        kind = row[0]
        if kind == 'hdr':
            ax.text(0.01, y, row[1], transform=ax.transAxes,
                    fontsize=7, color='#777', fontfamily='monospace', va='top')
        elif kind == 'kv':
            ax.text(0.03, y, row[1], transform=ax.transAxes,
                    fontsize=8, color='#333', va='top')
            ax.text(0.98, y, row[2], transform=ax.transAxes,
                    fontsize=8, color='#1a6fa8', va='top', ha='right',
                    fontfamily='monospace')
        elif kind == 'kvc':
            ax.text(0.03, y, row[1], transform=ax.transAxes,
                    fontsize=8, color='#333', va='top')
            ax.text(0.98, y, row[2], transform=ax.transAxes,
                    fontsize=8, color=row[3], va='top', ha='right',
                    fontfamily='monospace')
        elif kind == 'mono':
            ax.text(0.03, y, row[1], transform=ax.transAxes,
                    fontsize=7.5, color='#2c3e50', va='top',
                    fontfamily='monospace')
        elif kind == 'gap':
            y -= dy * 0.35
            continue
        y -= dy

    ax.set_title('Parameters', fontsize=10, fontweight='bold', pad=4)


# ─────────────────────────────────────────────────────────────────
# Main GUI
# ─────────────────────────────────────────────────────────────────

class CameraProjectionViewer:

    CAMERA_FIELDS = [
        ("Focal length (mm)", 'f_mm',     8.0),
        ("Sensor W (mm)",     'sensor_w', 8.8),
        ("Sensor H (mm)",     'sensor_h', 6.6),
        ("Width (px)",        'iw',       1944.0),
        ("Height (px)",       'ih',       1472.0),
        ("Near (mm)",         'z_near',   30.0),
        ("Far (mm)",          'z_far',    300.0),
    ]
    DOF_FIELDS = [
        ("Focus dist (mm)", 'focus_dist', 180.0),
    ]
    TAG_POS_FIELDS = [
        ("Tag Z depths (mm)",    'tag_z',       "198, 190, 167"),
        ("Lateral offsets (mm)", 'tag_lateral', "0, 31, 48"),
        ("Tag angle (deg)",      'tag_angle',   "0"),
    ]

    def __init__(self):
        self.current_lens     = "6 mm f/8"
        self.current_family   = list(APRILTAG_FAMILIES.keys())[0]
        # FIX: read tag size from the actual first family entry
        first_entry           = APRILTAG_FAMILIES[self.current_family][0]
        self.current_tag_lbl  = first_entry[0]
        self.current_tag_size = first_entry[1]
        self.show_dof         = True

        self._params = {}
        for _, k, v in self.CAMERA_FIELDS:
            self._params[k] = float(v)
        for _, k, v in self.DOF_FIELDS:
            self._params[k] = float(v)
        self._params['tag_z']       = "150"
        self._params['tag_lateral'] = "0"
        self._params['tag_angle']   = "0"
        # FIX: initialise from the actual family, not a hardcoded 40 mm
        self._params['tag_size']    = self.current_tag_size

        self._cone_cache      = ArtistCache()
        self._tag_cache       = ArtistCache()
        self._last_static_key = None
        self._last_tag_key    = None
        self._zone_patches    = []

        self.fig = plt.figure(figsize=(22, 12))
        self.fig.patch.set_facecolor('#f5f6fa')
        self.fig.canvas.manager.set_window_title(
            'Camera Projection Cone Viewer  v6')

        self._build_layout()
        self._build_section_labels()
        self._build_textboxes()
        self._build_dropdowns()
        self._build_misc_buttons()
        self.update(None)

    def _build_layout(self):
        gs = GridSpec(1, 3, figure=self.fig,
                      left=0.03, right=0.97, top=0.93, bottom=0.42,
                      wspace=0.32)
        self.ax_cone    = self.fig.add_subplot(gs[0, 0:2])
        self.ax_tag     = self.fig.add_subplot(gs[0, 2])
        self.ax_metrics = self.fig.add_axes([0.770, 0.018, 0.218, 0.368])
        self.ax_metrics.set_facecolor('#ffffff')
        for sp in self.ax_metrics.spines.values():
            sp.set_edgecolor('#cccccc')

    def _build_section_labels(self):
        ls = dict(fontsize=8.5, fontweight='bold', color='white',
                  va='center', ha='left')
        sections = [
            (0.030, 0.397, 0.175, '  Intrinsics & Depth',     '#2c3e50'),
            (0.215, 0.397, 0.110, '  Depth of Field',          '#00838f'),
            (0.215, 0.195, 0.110, '  Tag Position & Angle',    '#6a1b9a'),
            (0.335, 0.397, 0.425, '  Lens / Tag dropdowns (click to open)', '#37474f'),
        ]
        for x0, y0, w, lbl, col in sections:
            bar = self.fig.add_axes([x0, y0, w, 0.020])
            bar.set_facecolor(col)
            bar.axis('off')
            bar.text(0.01, 0.5, lbl, transform=bar.transAxes, **ls)

    def _build_textboxes(self):
        self.textboxes = {}
        col_xs  = [0.030, 0.120]
        tb_w, tb_h, row_step, lbl_off = 0.080, 0.024, 0.048, 0.026

        # Intrinsics + depth columns
        for i, (lbl, key, default) in enumerate(self.CAMERA_FIELDS):
            col = i // 4
            row = i %  4
            x0  = col_xs[col]
            y0  = 0.342 - row * row_step
            self.fig.text(x0, y0 + lbl_off, lbl,
                          fontsize=7, color='#333', va='bottom')
            ax_tb = self.fig.add_axes([x0, y0, tb_w, tb_h])
            tb = TextBox(ax_tb, '', initial=str(float(default)))
            tb.on_submit(lambda val, k=key: self._on_tb_float(k, val))
            self.textboxes[key] = tb

        # DoF
        dof_y = 0.342
        for i, (lbl, key, default) in enumerate(self.DOF_FIELDS):
            y0 = dof_y - i * row_step
            self.fig.text(0.215, y0 + lbl_off, lbl,
                          fontsize=7, color='#00838f', va='bottom')
            ax_tb = self.fig.add_axes([0.215, y0, 0.100, tb_h])
            ax_tb.set_facecolor('#e0f7fa')
            tb = TextBox(ax_tb, '', initial=str(float(default)))
            tb.on_submit(lambda val, k=key: self._on_tb_float(k, val))
            self.textboxes[key] = tb

        # Tag position + angle (stored as strings; first two support comma lists)
        tag_y = 0.165
        colors = {'tag_z': '#f3e5f5', 'tag_lateral': '#f3e5f5',
                  'tag_angle': '#ede7f6'}
        text_colors = {'tag_z': '#6a1b9a', 'tag_lateral': '#6a1b9a',
                       'tag_angle': '#6a1b9a'}
        for i, (lbl, key, default) in enumerate(self.TAG_POS_FIELDS):
            y0 = tag_y - i * row_step
            self.fig.text(0.215, y0 + lbl_off, lbl,
                          fontsize=7, color=text_colors.get(key, '#6a1b9a'),
                          va='bottom')
            ax_tb = self.fig.add_axes([0.215, y0, 0.100, tb_h])
            ax_tb.set_facecolor(colors.get(key, '#f3e5f5'))
            tb = TextBox(ax_tb, '', initial=str(default))
            tb.on_submit(lambda val, k=key: self._on_tb_str(k, val))
            self.textboxes[key] = tb

    def _on_tb_float(self, key, val):
        try:
            self._params[key] = float(val)
        except ValueError:
            pass
        self.update(None)

    def _on_tb_str(self, key, val):
        self._params[key] = val
        self.update(None)

    def _build_dropdowns(self):
        lens_names  = list(LENS_TYPES.keys())
        fam_names   = list(APRILTAG_FAMILIES.keys())
        size_labels = [lbl for lbl, _ in APRILTAG_FAMILIES[self.current_family]]

        def ph(n): return min(0.022 * n + 0.01, 0.38)

        self.dd_lens = DropdownMenu(
            self.fig,
            btn_rect=[0.335, 0.350, 0.185, 0.030],
            panel_rect=[0.335, 0.420, 0.185, ph(len(lens_names))],
            options=lens_names, initial=self.current_lens,
            on_select=self._select_lens,
            btn_color='#e8f0fe', active_color='#1565c0',
            label_fontsize=7.5, header_color='#1565c0',
        )
        self.fig.text(0.335, 0.383, 'Lens (f-number)',
                      fontsize=7.5, fontweight='bold', color='#1565c0')

        self.dd_family = DropdownMenu(
            self.fig,
            btn_rect=[0.530, 0.350, 0.145, 0.030],
            panel_rect=[0.530, 0.420, 0.145, ph(len(fam_names))],
            options=fam_names, initial=self.current_family,
            on_select=self._select_family,
            btn_color='#fff3e0', active_color='#e65100',
            label_fontsize=7.5, header_color='#e65100',
        )
        self.fig.text(0.530, 0.383, 'Tag family',
                      fontsize=7.5, fontweight='bold', color='#e65100')

        self.dd_size = DropdownMenu(
            self.fig,
            btn_rect=[0.685, 0.350, 0.075, 0.030],
            panel_rect=[0.685, 0.420, 0.075, ph(len(size_labels))],
            # FIX: initial matches the actual first entry, not hardcoded
            options=size_labels, initial=self.current_tag_lbl,
            on_select=self._select_tag_size,
            btn_color='#e8f5e9', active_color='#2e7d32',
            label_fontsize=7.5, header_color='#2e7d32',
        )
        self.fig.text(0.685, 0.383, 'Tag size',
                      fontsize=7.5, fontweight='bold', color='#2e7d32')

    def _select_lens(self, label):
        self.current_lens    = label
        f_mm_new             = LENS_TYPES[label]['f_mm']
        self._params['f_mm'] = f_mm_new
        # FIX: update the focal-length textbox so it reflects the dropdown choice
        if 'f_mm' in self.textboxes:
            self.textboxes['f_mm'].set_val(str(f_mm_new))
        self._last_static_key = None
        self._last_tag_key    = None
        self.update(None)

    def _select_family(self, name):
        self.current_family      = name
        first                    = APRILTAG_FAMILIES[name][0]
        self.current_tag_lbl     = first[0]
        self.current_tag_size    = first[1]
        self._params['tag_size'] = self.current_tag_size
        self.dd_size.set_options([lbl for lbl, _ in APRILTAG_FAMILIES[name]],
                                 new_selection=self.current_tag_lbl)
        self._last_static_key = None
        self._last_tag_key    = None
        self.update(None)

    def _select_tag_size(self, label):
        for lbl, sz in APRILTAG_FAMILIES[self.current_family]:
            if lbl == label:
                self.current_tag_lbl     = lbl
                self.current_tag_size    = sz
                self._params['tag_size'] = sz
                break
        self._last_static_key = None
        self._last_tag_key    = None
        self.update(None)

    def _build_misc_buttons(self):
        bs = dict(color='#ecf0f1', hovercolor='#bdc3c7')
        for rect, lbl, cb in [
            ([0.030, 0.025, 0.080, 0.024], 'Print K matrix', self._print_k),
            ([0.120, 0.025, 0.075, 0.024], 'Export JSON',    self._export_json),
            ([0.205, 0.025, 0.085, 0.024], 'Toggle DoF',     self._toggle_dof),
        ]:
            ax_b = self.fig.add_axes(rect)
            btn  = Button(ax_b, lbl, **bs)
            btn.label.set_fontsize(7.5)
            btn.on_clicked(cb)

        self.fig.text(
            0.5, 0.975,
            'Camera Projection Cone Viewer  v6  ·  mm scale  ·  DoF + AprilTag  ·  multi-tag  ·  tag angle',
            ha='center', va='top', fontsize=11, fontweight='bold', color='#2c3e50')

    def _toggle_dof(self, _=None):
        self.show_dof = not self.show_dof
        self.update(None)

    def _print_k(self, _=None):
        p = self._params
        fx, fy, cx, cy = compute_intrinsics(
            p['f_mm'], p['sensor_w'], p['sensor_h'], p['iw'], p['ih'])
        K  = get_K_matrix(fx, fy, cx, cy)
        hf, vf, df = compute_fov(fx, fy, p['iw'], p['ih'])
        angle = self._get_angle()
        eff   = effective_tag_size(p['tag_size'], angle)
        print("\n── Camera K matrix ──────────────────────────────────")
        print(f"  Lens   : {self.current_lens}")
        print(f"  Family : {self.current_family}  |  Tag: {self.current_tag_lbl}")
        print(f"  Tag angle: {angle:.1f}°  |  Eff. size: {eff:.2f} mm")
        print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
        print(f"  Image  : {int(p['iw'])}×{int(p['ih'])} px")
        print(f"  H-FOV  : {hf:.2f}°   V-FOV: {vf:.2f}°   Diag: {df:.2f}°")
        print(f"\n  K =\n{K}\n")
        print("──────────────────────────────────────────────────────\n")

    def _export_json(self, _=None):
        p = self._params
        fx, fy, cx, cy = compute_intrinsics(
            p['f_mm'], p['sensor_w'], p['sensor_h'], p['iw'], p['ih'])
        hfov, vfov, diag = compute_fov(fx, fy, p['iw'], p['ih'])
        K = get_K_matrix(fx, fy, cx, cy)
        dof_n, dof_f, hyp = self._calc_dof(fx)
        tags  = self._parse_tags()
        angle = self._get_angle()
        eff   = effective_tag_size(p['tag_size'], angle)
        config = {
            "lens": self.current_lens,
            "f_number": LENS_TYPES[self.current_lens]['f_number'],
            "camera_geometry": {
                "focal_length_mm":  p['f_mm'],
                "sensor_width_mm":  p['sensor_w'],
                "sensor_height_mm": p['sensor_h'],
                "image_width_px":   p['iw'],
                "image_height_px":  p['ih'],
            },
            "intrinsics": {"fx": round(fx,3), "fy": round(fy,3),
                           "cx": round(cx,3), "cy": round(cy,3)},
            "fov": {"h_deg": round(hfov,3), "v_deg": round(vfov,3),
                    "diag_deg": round(diag,3)},
            "depth_range_mm": {"near": p['z_near'], "far": p['z_far']},
            "dof": {
                "focus_dist_mm": p['focus_dist'],
                "dof_near_mm":   round(dof_n, 1) if dof_n else None,
                "dof_far_mm":    round(dof_f, 1) if dof_f != float('inf') else "inf",
                "hyperfocal_mm": round(hyp,  1) if hyp  != float('inf') else "inf",
            },
            "apriltag": {
                "family":            self.current_family,
                "size_mm":           p['tag_size'],
                "angle_deg":         angle,
                "effective_size_mm": round(eff, 3),
                "detection_distances_mm": {
                    lbl: round(detection_distance(fx, eff, px), 1)
                    for px, _, lbl in DETECTION_THRESHOLDS
                },
            },
            "tag_positions": [
                {"z_mm": tz, "lateral_mm": tl,
                 "projected_px": round(fx * eff / tz, 2) if tz > 0 and eff > 0 else None}
                for tz, tl in tags
            ],
            "K_matrix": K.tolist(),
        }
        fname = "camera_config.json"
        with open(fname, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n── Exported config → {fname}\n")

    # ── Helpers ───────────────────────────────────────────────────

    def _calc_dof(self, fx_eff):
        if not self.show_dof:
            return None, None, None
        p   = self._params
        li  = LENS_TYPES[self.current_lens]
        psz = pixel_size_mm(p['sensor_w'], p['iw'])
        coc = auto_coc(p['sensor_w'], p['sensor_h'])
        return compute_dof(fx_eff * psz, li['f_number'], p['focus_dist'], coc)

    def _parse_tags(self):
        zs   = parse_float_list(self._params.get('tag_z', ''))
        lats = parse_float_list(self._params.get('tag_lateral', ''))
        if not zs:
            return []
        while len(lats) < len(zs):
            lats.append(0.0)
        return list(zip(zs, lats[:len(zs)]))

    def _get_angle(self):
        """Tag yaw angle in degrees, clamped to [−89.9, 89.9]."""
        vals = parse_float_list(str(self._params.get('tag_angle', '0')))
        a    = vals[0] if vals else 0.0
        return float(np.clip(a, -89.9, 89.9))

    # ── Main update ───────────────────────────────────────────────

    def update(self, _):
        for dd in DropdownMenu._registry:
            if dd._open:
                dd.close()

        p  = self._params
        fx, fy, cx, cy = compute_intrinsics(
            p['f_mm'], p['sensor_w'], p['sensor_h'], p['iw'], p['ih'])
        iw, ih = p['iw'], p['ih']
        zn     = max(1.0, p['z_near'])
        zf     = max(zn + 1.0, p['z_far'])
        tag    = p['tag_size']
        tags   = self._parse_tags()
        angle  = self._get_angle()
        eff    = effective_tag_size(tag, angle)

        dof_n, dof_f, hyp = self._calc_dof(fx)

        static_key = (round(fx, 3), round(iw), round(zn), round(zf),
                      round(tag, 2), round(angle, 2))
        tag_key    = (round(fx, 3), round(tag, 2), round(eff, 3), round(zf),
                      self.current_family, self.current_tag_lbl)

        if static_key != self._last_static_key:
            self.ax_cone.cla()
            self._cone_cache.clear_all()
            self._zone_patches = build_cone_static(
                self.ax_cone, self._cone_cache,
                fx, iw, zn, zf, tag, eff)
            self._last_static_key = static_key

        if tag_key != self._last_tag_key:
            self.ax_tag.cla()
            self._tag_cache.clear_all()
            build_tag_static(
                self.ax_tag, self._tag_cache,
                fx, tag, eff, zf,
                self.current_family, self.current_tag_lbl, angle)
            self._last_tag_key = tag_key

        update_cone_dynamic(
            self.ax_cone, self._cone_cache,
            fx, iw, zf, dof_n, dof_f, tags, eff,
            self._zone_patches)

        update_tag_dynamic(
            self.ax_tag, self._tag_cache,
            fx, eff, zf, tags, dof_n, dof_f)

        draw_metrics(
            self.ax_metrics,
            fx, fy, cx, cy, iw, ih,
            zn, zf, tag, eff, angle,
            self.current_lens, self.current_family, self.current_tag_lbl,
            dof_near=dof_n, dof_far=dof_f, hyperfocal=hyp,
            focus_dist=p['focus_dist'] if self.show_dof else None,
            tags=tags)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    viewer = CameraProjectionViewer()
    viewer.show()
