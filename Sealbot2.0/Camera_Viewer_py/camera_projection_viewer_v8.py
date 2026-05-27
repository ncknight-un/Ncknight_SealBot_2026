"""
Camera Projection Cone Viewer  v8
===================================
Interactive GUI for visualizing camera projection cones, depth-of-field,
and AprilTag detection ranges.

Units: millimeters (mm)

Changes in v8:
    - Detection-range plot removed; metrics panel now full height.
    - Export JSON removed.
    - Dropdowns stacked in a single column on the left of the control strip.
    - FPS + ROI inputs moved up into the camera-intrinsics section.
    - H/V FOV toggle button: switches cone plot between horizontal and
      vertical cross-section view.
    - Metrics panel is wide enough to display all bandwidth rows.
"""

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons, TextBox

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*')

# ─────────────────────────────────────────────────────────────────
# Catalogues
# ─────────────────────────────────────────────────────────────────
LENS_TYPES = {
    "2.0 mm f/1.2":  dict(f_mm=2.0,  f_number=1.2),
    "2.0 mm f/1.4":  dict(f_mm=2.0,  f_number=1.4),
    "2.0 mm f/1.8":  dict(f_mm=2.0,  f_number=1.8),
    "5 mm f/1.2":    dict(f_mm=5.0,  f_number=1.2),
    "6 mm f/1.8":    dict(f_mm=6.0,  f_number=1.8),
    "6 mm f/4":      dict(f_mm=6.0,  f_number=4.0),
    "6 mm f/8":      dict(f_mm=6.0,  f_number=8.0),
    "6 mm f/16":     dict(f_mm=6.0,  f_number=16.0),
    "8 mm f/1.4":    dict(f_mm=8.0,  f_number=1.4),
    "8 mm f/1.8":    dict(f_mm=8.0,  f_number=1.8),
    "8.5 mm f/4.0":  dict(f_mm=8.5,  f_number=4.0),
    "8.5 mm f/5.6":  dict(f_mm=8.5,  f_number=5.5),
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

PIXEL_FORMATS = {
    "Mono8":    1.0,
    "Bayer8":   1.0,
    "Bayer10p": 1.25,
    "Bayer10":  2.0,
    "Bayer12p": 1.5,
    "Bayer12":  2.0,
    "Mono10p":  1.25,
    "Mono10":   2.0,
    "Mono12p":  1.5,
    "Mono12":   2.0,
    "Mono16":   2.0,
    "YUV422":   2.0,
    "RGB8":     3.0,
    "BGR8":     3.0,
}

INTERFACES = [
    ("USB 2.0",        60),
    ("GigE",          125),
    ("2.5 GigE",      312),
    ("USB 3.0",       400),
    ("MIPI CSI-2 2L", 450),
    ("CoaXPress x1",  750),
    ("USB 3.1 Gen2",  900),
    ("MIPI CSI-2 4L", 900),
    ("10 GigE",      1250),
    ("CoaXPress x2", 1500),
    ("CoaXPress x4", 3000),
]

APRILTAG_FAMILIES = {
    "Tag36h11": [
        ("5 mm",   5.0),  ("10 mm", 10.0),  ("20 mm",  20.0), ("40 mm",  40.0),
        ("60 mm",  60.0), ("80 mm", 80.0),  ("120 mm", 120.0),("160 mm", 160.0),
        ("200 mm",200.0), ("250 mm",250.0), ("300 mm", 300.0),
    ],
    "Tag25h9": [
        ("5 mm",   5.0),  ("10 mm", 10.0),  ("15 mm",  15.0), ("30 mm",  30.0),
        ("60 mm",  60.0), ("90 mm", 90.0),  ("120 mm",120.0), ("180 mm",180.0),
        ("240 mm",240.0), ("300 mm",300.0),
    ],
    "Tag16h5": [
        ("5 mm",   5.0),  ("10 mm", 10.0),  ("20 mm",  20.0), ("40 mm",  40.0),
        ("60 mm",  60.0), ("100 mm",100.0), ("150 mm",150.0), ("200 mm",200.0),
    ],
    "TagStd41h12": [
        ("10 mm",  10.0), ("20 mm", 20.0),  ("30 mm",  30.0), ("60 mm",  60.0),
        ("100 mm",100.0), ("150 mm",150.0), ("200 mm",200.0), ("300 mm",300.0),
    ],
    "Circle21h7": [
        ("10 mm",  10.0), ("20 mm", 20.0),  ("50 mm",  50.0), ("80 mm",  80.0),
        ("100 mm",100.0), ("150 mm",150.0), ("200 mm",200.0),
    ],
}

DETECTION_THRESHOLDS = [
    (10, '#e67e22', 'Poor (10 px)'),
    (20, '#f1c40f', 'Fair (20 px)'),
    (40, '#2ecc71', 'Good (40 px)'),
    (80, '#3498db', 'Excellent (80 px)'),
]

TAG_COLORS = ['#8e44ad','#c0392b','#16a085','#d35400',
              '#2980b9','#27ae60','#8e44ad','#f39c12']


# ─────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────

def effective_tag_size(tag_size_mm, angle_deg):
    return tag_size_mm * abs(np.cos(np.radians(float(angle_deg))))

def compute_fov(fx, fy, iw, ih):
    hfov = np.degrees(2 * np.arctan(iw / (2 * fx)))
    vfov = np.degrees(2 * np.arctan(ih / (2 * fy)))
    diag = np.degrees(2 * np.arctan(
        np.sqrt(iw**2 + ih**2) / (2 * np.sqrt(fx * fy))))
    return hfov, vfov, diag

def frustum_half(f, dim, z):
    """Half-width (horizontal) or half-height (vertical) at depth z."""
    return z * dim / (2 * f)

def get_K_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=float)

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

def auto_coc(sw, sh):
    return np.sqrt(sw**2 + sh**2) / 1500.0

def compute_dof(fx_mm_equiv, f_number, focus_dist_mm, coc_mm):
    f, N, c = fx_mm_equiv, f_number, coc_mm
    if c <= 0 or f <= 0 or focus_dist_mm <= 0:
        return 0.0, float('inf'), float('inf')
    H    = (f*f) / (N*c) + f
    s    = focus_dist_mm
    near = (s*(H-f)) / (H+s-2*f) if (H+s-2*f) > 0 else s/2
    far  = (s*(H-f)) / (H-f-s)   if (H-f-s)   > 0 else float('inf')
    if s >= H: far = float('inf')
    return max(1.0, near), far, H

def parse_float_list(text):
    out = []
    for tok in str(text).split(','):
        tok = tok.strip()
        if tok:
            try: out.append(float(tok))
            except ValueError: pass
    return out

def compute_bandwidth(iw, ih, fps, pixel_format, roi_x, roi_y, roi_w, roi_h, scale):
    iw, ih = max(1, int(iw)), max(1, int(ih))
    fps    = max(0.0, float(fps))
    bpp    = PIXEL_FORMATS.get(pixel_format, 1.0)
    scale  = float(np.clip(scale, 0.01, 1.0))
    rx = int(np.clip(roi_x, 0, iw-1))
    ry = int(np.clip(roi_y, 0, ih-1))
    rw = int(np.clip(roi_w, 1, iw-rx))
    rh = int(np.clip(roi_h, 1, ih-ry))
    out_w = max(1, int(rw*scale) & ~1)
    out_h = max(1, int(rh*scale) & ~1)
    pixels   = out_w * out_h
    bpf      = pixels * bpp
    bps      = bpf * fps
    raw_mb_s = bps / 1e6
    raw_mbps = bps * 8 / 1e6
    raw_gbps = bps * 8 / 1e9
    return dict(
        sensor_w=iw, sensor_h=ih,
        roi_x=rx, roi_y=ry, roi_w=rw, roi_h=rh,
        out_w=out_w, out_h=out_h,
        pixels_per_frame=pixels, bytes_per_frame=bpf,
        fps=fps, pixel_format=pixel_format, bpp=bpp, scale=scale,
        raw_mb_s=raw_mb_s, raw_mbps=raw_mbps, raw_gbps=raw_gbps,
        comp_mjpeg=raw_mb_s/5,
        comp_h264 =raw_mb_s/10,
        comp_h265 =raw_mb_s/20,
    )


# ─────────────────────────────────────────────────────────────────
# Dropdown widget
# ─────────────────────────────────────────────────────────────────

class DropdownMenu:
    _registry: list = []

    def __init__(self, fig, btn_rect, panel_rect, options,
                 initial=None, on_select=None,
                 btn_color='#ecf0f1', active_color='#2c3e50',
                 label_fontsize=7.5, header_color='#2c3e50'):
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
    def _trunc(text, n=24):
        return text if len(text) <= n else text[:n-1]+'…'

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
        idx = self.options.index(sel)
        self._radio = RadioButtons(self._ax_panel, self.options,
                                   active=idx, activecolor=self._radio.activecolor)
        for lbl in self._radio.labels:
            lbl.set_fontsize(7.5)
        self._radio.on_clicked(self._on_select)
        self.close()

    @property
    def value(self): return self._current


# ─────────────────────────────────────────────────────────────────
# Artist cache
# ─────────────────────────────────────────────────────────────────

class ArtistCache:
    def __init__(self): self._g: dict = {}
    def add(self, g, a): self._g.setdefault(g, []).append(a)
    def clear(self, *groups):
        for g in (groups or list(self._g.keys())):
            for a in self._g.pop(g, []):
                try: a.remove()
                except: pass
    def clear_all(self): self.clear(*list(self._g.keys()))


# ─────────────────────────────────────────────────────────────────
# Cone plot — static geometry
# ─────────────────────────────────────────────────────────────────

def build_cone_static(ax, cache, fx, fy, iw, ih,
                      z_near, z_far, tag_size_mm, show_vertical):
    """
    Draw frustum, detection zones, FOV arc.
    show_vertical=True  →  vertical cross-section (uses fy, ih)
    show_vertical=False →  horizontal cross-section (uses fx, iw)
    """
    cache.clear('cone_static')

    if show_vertical:
        f_use, dim_use = fy, ih
        fov_label = 'V-FOV'
        axis_label = 'Vertical extent (mm)'
    else:
        f_use, dim_use = fx, iw
        fov_label = 'H-FOV'
        axis_label = 'Horizontal extent (mm)'

    fov_rad = 2 * np.arctan(dim_use / (2 * f_use))
    z_vals  = np.linspace(0, z_far, 800)
    half_e  = z_vals * np.tan(fov_rad / 2)
    hf      = z_far  * np.tan(fov_rad / 2)

    def add(a):
        cache.add('cone_static', a)
        return a

    add(ax.fill_between(z_vals, -half_e, half_e,
                        alpha=0.10, color='steelblue', zorder=1))
    add(ax.plot(z_vals,  half_e, color='steelblue', lw=2.0,
                label='Frustum boundary')[0])
    add(ax.plot(z_vals, -half_e, color='steelblue', lw=2.0)[0])
    add(ax.axhline(0, color='gray', lw=0.8, ls=':', alpha=0.4, zorder=1))

    zone_patches = []
    if tag_size_mm > 0:
        prev_z = z_near
        for min_px, color, lbl in DETECTION_THRESHOLDS:
            z_t = min(detection_distance(f_use, tag_size_mm, min_px), z_far)
            if z_t > prev_z:
                add(ax.axvspan(prev_z, z_t, alpha=0.06, color=color, zorder=1))
                add(ax.axvline(z_t, color=color, lw=1.2, ls='--',
                               alpha=0.80, zorder=2))
                add(ax.text(z_t, hf*0.88, f'{min_px}px\n{z_t:.0f}',
                            fontsize=6.5, color=color, ha='center', va='top',
                            bbox=dict(fc='white', ec='none', alpha=0.5, pad=1)))
            prev_z = z_t
            zone_patches.append(mpatches.Patch(color=color, alpha=0.5, label=lbl))
            if z_t >= z_far: break

    arc_r = z_far * 0.13
    theta  = np.linspace(-fov_rad/2, fov_rad/2, 120)
    add(ax.plot([0, arc_r*np.cos(fov_rad/2)], [0,  arc_r*np.sin(fov_rad/2)],
                color='#f39c12', lw=1.3, alpha=0.8)[0])
    add(ax.plot([0, arc_r*np.cos(fov_rad/2)], [0, -arc_r*np.sin(fov_rad/2)],
                color='#f39c12', lw=1.3, alpha=0.8)[0])
    add(ax.plot(arc_r*np.cos(theta), arc_r*np.sin(theta),
                color='#f39c12', lw=2.0)[0])
    add(ax.annotate(f'{fov_label}\n{np.degrees(fov_rad):.1f}°',
                    xy=(arc_r*1.12, 0), fontsize=8,
                    color='#f39c12', va='center', ha='left'))
    add(ax.scatter([0],[0], s=80, color='steelblue', zorder=6,
                   label='Camera origin'))

    title_suffix = ' (Vertical)' if show_vertical else ' (Horizontal)'
    ax.set_xlabel('Depth Z (mm)', fontsize=9)
    ax.set_ylabel(axis_label, fontsize=9)
    ax.set_title(f'Top-down FOV cone{title_suffix}',
                 fontsize=10, fontweight='bold', pad=4)
    ax.set_xlim(0, z_far * 1.10)
    ax.grid(True, alpha=0.18)
    return zone_patches, f_use, fov_rad


def update_cone_dynamic(ax, cache, f_use, fov_rad, z_far,
                        dof_near, dof_far,
                        tags, zone_patches, show_vertical):
    cache.clear('cone_dynamic')

    def add(a):
        cache.add('cone_dynamic', a)
        return a

    if dof_near is not None and dof_far is not None:
        dn = float(np.clip(dof_near, 0, z_far*1.05))
        df = (float(np.clip(dof_far, 0, z_far*1.05))
              if dof_far != float('inf') else z_far*1.05)
        z_dof  = np.linspace(dn, df, 400)
        hw_dof = z_dof * np.tan(fov_rad/2)
        far_lbl = '∞' if dof_far == float('inf') else f'{dof_far:.0f}'
        add(ax.fill_between(z_dof, -hw_dof, hw_dof,
                            alpha=0.22, color='#00bcd4', zorder=2,
                            label=f'DoF  {dn:.0f}–{far_lbl} mm'))
        add(ax.axvline(dn, color='#00bcd4', lw=2.0, ls='-.', alpha=0.9,
                       zorder=3, label=f'DoF near  {dn:.0f} mm'))
        if dof_far != float('inf') and df <= z_far*1.05:
            add(ax.axvline(df, color='#0097a7', lw=2.0, ls='-.', alpha=0.9,
                           zorder=3, label=f'DoF far  {df:.0f} mm'))
        else:
            add(ax.axvline(z_far*1.04, color='#0097a7', lw=1.5, ls=':',
                           alpha=0.5, zorder=3, label='DoF far → ∞'))

    for i, (tz, tlat, t_angle, t_eff) in enumerate(tags):
        if tz <= 0: continue
        color   = TAG_COLORS[i % len(TAG_COLORS)]
        hw_at_z = tz * np.tan(fov_rad/2)
        in_fov  = abs(tlat) <= hw_at_z
        mcol    = color if in_fov else '#c0392b'
        px_size = f_use * t_eff / tz if t_eff > 0 else None

        add(ax.scatter([tz], [tlat], s=160, marker='*', color=mcol, zorder=8,
                       label=f'Tag {i+1} ({tz:.0f},{tlat:.0f}) mm'))
        fov_str   = 'in FOV' if in_fov else 'OUT'
        angle_str = f' {t_angle:.0f}°' if abs(t_angle) > 0.5 else ''
        px_str    = f'{px_size:.1f} px' if px_size is not None else ''
        add(ax.annotate(f'T{i+1}{angle_str}: {fov_str}\n{px_str}',
                        xy=(tz, tlat),
                        xytext=(tz + z_far*0.04, tlat + hw_at_z*0.15),
                        fontsize=8, color=mcol, va='bottom', ha='left',
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=mcol, lw=1.2),
                        bbox=dict(fc='white', ec=mcol, alpha=0.85,
                                  pad=2, boxstyle='round')))

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + zone_patches,
              loc='upper left', fontsize=7, framealpha=0.70,
              ncol=2, columnspacing=0.8, handlelength=1.5)


# ─────────────────────────────────────────────────────────────────
# Metrics panel
# ─────────────────────────────────────────────────────────────────

def draw_metrics(ax, fx, fy, cx, cy, iw, ih,
                 z_near, z_far, tag_size,
                 lens_name, family, tag_label,
                 tags, bw, show_vertical,
                 dof_near=None, dof_far=None, hyperfocal=None,
                 focus_dist=None):
    ax.clear()
    ax.axis('off')

    hfov, vfov, diag = compute_fov(fx, fy, iw, ih)
    li = LENS_TYPES[lens_name]

    # Frustum extents at near/far using the displayed axis
    f_use  = fy if show_vertical else fx
    dim_use = ih if show_vertical else iw
    hw_n   = frustum_half(f_use, dim_use, z_near)
    hw_f   = frustum_half(f_use, dim_use, z_far)
    ext_lbl = 'V extent' if show_vertical else 'H extent'

    rows = [
        ('hdr', '── Lens ──────────────────────────'),
        ('kv',  'Focal length', f"{li['f_mm']:.1f} mm"),
        ('kv',  'f-number',     f"f/{li['f_number']}"),
        ('gap', ''),
        ('hdr', '── Intrinsics ────────────────────'),
        ('kv',  'fx / fy',    f"{fx:.0f} / {fy:.0f} px"),
        ('kv',  'cx / cy',    f"{cx:.0f} / {cy:.0f} px"),
        ('kv',  'Sensor res', f"{int(iw)} × {int(ih)} px"),
        ('gap', ''),
        ('hdr', '── Field of View ─────────────────'),
        ('kv',  'H-FOV',    f"{hfov:.1f}°"),
        ('kv',  'V-FOV',    f"{vfov:.1f}°"),
        ('kv',  'Diag FOV', f"{diag:.1f}°"),
        ('gap', ''),
        ('hdr', f'── {ext_lbl} at depth ────────────'),
        ('kv',  f'@ {z_near:.0f} mm', f"{2*hw_n:.1f} mm"),
        ('kv',  f'@ {z_far:.0f} mm',  f"{2*hw_f:.1f} mm"),
    ]

    if dof_near is not None:
        far_str = '∞' if dof_far == float('inf') else f'{dof_far:.0f} mm'
        hyp_str = '∞' if hyperfocal == float('inf') else f'{hyperfocal:.0f} mm'
        rows += [
            ('gap', ''),
            ('hdr', '── Depth of Field ────────────────'),
            ('kv',  'Focus dist', f"{focus_dist:.0f} mm" if focus_dist else '—'),
            ('kvc', 'DoF near',   f"{dof_near:.0f} mm", '#00bcd4'),
            ('kvc', 'DoF far',    far_str,              '#0097a7'),
            ('kv',  'Hyperfocal', hyp_str),
            ('kv',  'DoF span',   ('∞' if dof_far==float('inf')
                                   else f'{dof_far-dof_near:.0f} mm')),
        ]

    if tag_size > 0:
        rows += [('gap',''), ('hdr', f'── {family}  {tag_label} ────────')]
        ref_eff = tag_size
        if tags:
            _, _, a0, e0 = tags[0]
            if abs(a0) > 0.5:
                rows.append(('kvc','T1 angle',
                              f'{a0:.1f}° → eff {e0:.1f} mm','#9b59b6'))
            ref_eff = e0
        for min_px, color, lbl in DETECTION_THRESHOLDS:
            z_t = detection_distance(f_use, ref_eff, min_px)
            rows.append(('kvc', lbl,
                          f"{z_t:.0f} mm" if z_t < 1e6 else '> range',
                          color))

    if tags:
        rows += [('gap',''), ('hdr','── Tag positions ─────────────────')]
        for i, (tz, tlat, t_angle, t_eff) in enumerate(tags):
            if tz > 0 and t_eff > 0:
                px = f_use * t_eff / tz
                astr = f' {t_angle:.0f}°' if abs(t_angle) > 0.5 else ''
                rows.append(('kvc',
                              f'T{i+1}{astr} ({tz:.0f},{tlat:.0f})',
                              f'{px:.1f} px',
                              TAG_COLORS[i % len(TAG_COLORS)]))

    # Bandwidth
    if bw:
        roi_full = (bw['roi_x']==0 and bw['roi_y']==0
                    and bw['roi_w']==bw['sensor_w']
                    and bw['roi_h']==bw['sensor_h'])
        roi_str  = ('full sensor' if roi_full
                    else f"{bw['roi_w']}×{bw['roi_h']} @({bw['roi_x']},{bw['roi_y']})")
        scale_str = '1:1' if abs(bw['scale']-1)<0.001 else f"×{bw['scale']:.3f}"
        fitting   = [n for n, m in INTERFACES if m >= bw['raw_mb_s']]
        iface_str = fitting[0] if fitting else 'none — exceeds all!'
        iface_col = '#27ae60' if fitting else '#c0392b'

        rows += [
            ('gap',''),
            ('hdr','── Stream & Bandwidth ────────────'),
            ('kv', 'Pixel format', f"{bw['pixel_format']}  ({bw['bpp']} B/px)"),
            ('kv', 'ROI',          roi_str),
            ('kv', 'Scale',        scale_str),
            ('kv', 'Output res',   f"{bw['out_w']} × {bw['out_h']} px"),
            ('kv', 'FPS',          f"{bw['fps']:.1f}"),
            ('gap',''),
            ('hdr','── Raw (uncompressed) ────────────'),
            ('kvc','Bytes / frame', f"{bw['bytes_per_frame']:,.0f}", '#555555'),
            ('kvc','MB/s',          f"{bw['raw_mb_s']:.2f}",         '#c0392b'),
            ('kvc','Mbps',          f"{bw['raw_mbps']:.1f}",         '#c0392b'),
            ('kvc','Gbps',          f"{bw['raw_gbps']:.3f}",         '#c0392b'),
            ('gap',''),
            ('hdr','── Compressed estimates ──────────'),
            ('kvc','MJPEG  (~5:1)', f"{bw['comp_mjpeg']:.2f} MB/s",  '#e67e22'),
            ('kvc','H.264 (~10:1)', f"{bw['comp_h264']:.2f} MB/s",   '#f1c40f'),
            ('kvc','H.265 (~20:1)', f"{bw['comp_h265']:.2f} MB/s",   '#27ae60'),
            ('gap',''),
            ('hdr','── Interface fit ─────────────────'),
            ('kvc','Min. interface', iface_str, iface_col),
        ]

    rows += [
        ('gap',''),
        ('hdr','── K matrix ──────────────────────'),
        ('mono', f" [[{fx:.0f},   0,  {cx:.0f}]"),
        ('mono', f"  [  0,  {fy:.0f},  {cy:.0f}]"),
        ('mono', f"  [  0,    0,      1]]"),
    ]

    y, dy = 0.99, 0.031
    for row in rows:
        k = row[0]
        if k == 'hdr':
            ax.text(0.01, y, row[1], transform=ax.transAxes,
                    fontsize=6.3, color='#888', fontfamily='monospace', va='top')
        elif k == 'kv':
            ax.text(0.02, y, row[1], transform=ax.transAxes,
                    fontsize=7.0, color='#333', va='top')
            ax.text(0.99, y, row[2], transform=ax.transAxes,
                    fontsize=7.0, color='#1a6fa8', va='top', ha='right',
                    fontfamily='monospace')
        elif k == 'kvc':
            ax.text(0.02, y, row[1], transform=ax.transAxes,
                    fontsize=7.0, color='#333', va='top')
            ax.text(0.99, y, row[2], transform=ax.transAxes,
                    fontsize=7.0, color=row[3], va='top', ha='right',
                    fontfamily='monospace')
        elif k == 'mono':
            ax.text(0.02, y, row[1], transform=ax.transAxes,
                    fontsize=6.8, color='#2c3e50', va='top',
                    fontfamily='monospace')
        if k == 'gap':
            y -= dy * 0.38
        else:
            y -= dy

    ax.set_title('Parameters & Outputs', fontsize=10, fontweight='bold', pad=4)


# ─────────────────────────────────────────────────────────────────
# Main GUI
# ─────────────────────────────────────────────────────────────────

class CameraProjectionViewer:
    """
    Layout (figure coordinates, bottom=0 top=1):

    ┌─────────────────────────────────────────────────────────────┐
    │  title bar                                                   │
    ├───────────────────────────┬─────────────────────────────────┤
    │  ax_cone  (FOV plot)      │  ax_metrics (parameters panel)  │
    │  left=0.03 right=0.57     │  left=0.60  right=0.99          │
    │  top=0.93  bottom=0.42    │  top=0.93   bottom=0.02         │
    ├──────────┬────────────────┴─────────────────────────────────┤
    │ dropdowns│  textbox controls                                 │
    │ column   │  col-A: intrinsics+FPS+ROI  col-B: DoF  col-C: tag│
    └──────────┴──────────────────────────────────────────────────┘
    """

    # Camera intrinsics + depth + FPS + ROI all in one section
    CAMERA_FIELDS = [
        # label,             key,       default
        ("Focal len (mm)",  'f_mm',     6.0),
        ("Sensor W (mm)",   'sensor_w', 6.8),
        ("Sensor H (mm)",   'sensor_h', 5.7),
        ("Width (px)",      'iw',       2464.0),
        ("Height (px)",     'ih',       2064.0),
        ("Near (mm)",       'z_near',   30.0),
        ("Far (mm)",        'z_far',    300.0),
        ("FPS",             'fps',      30.0),
        ("ROI x0 (px)",     'roi_x',    0.0),
        ("ROI y0 (px)",     'roi_y',    0.0),
        ("ROI w (px)",      'roi_w',    2464.0),
        ("ROI h (px)",      'roi_h',    2064.0),
        ("Scale factor",    'scale',    1.0),
    ]
    DOF_FIELDS = [
        ("Focus dist (mm)", 'focus_dist', 150.0),
    ]
    TAG_POS_FIELDS = [
        ("Tag Z (mm)",      'tag_z',       "150"),
        ("Lateral (mm)",    'tag_lateral', "0"),
        ("Angles (deg)",    'tag_angle',   "0"),
    ]

    # Layout constants
    DD_X      = 0.030   # dropdown column x
    DD_W      = 0.095   # dropdown button width
    DD_H      = 0.028   # dropdown button height
    DD_GAP    = 0.038   # vertical gap between dropdown buttons
    DD_TOP    = 0.375   # y of first dropdown button

    TB_H      = 0.024   # textbox height
    ROW_STEP  = 0.046   # vertical step between textbox rows
    LBL_OFF   = 0.025   # label above textbox

    # Textbox columns start x (to the right of dropdown column)
    COL_A_X   = 0.140   # intrinsics + stream
    COL_B_X   = 0.390   # DoF + tag position
    COL_C_X   = 0.510   # (unused — merged into col-B with a gap)
    TB_W      = 0.090   # textbox width

    def __init__(self):
        self.current_lens      = "6 mm f/8"
        self.current_family    = list(APRILTAG_FAMILIES.keys())[0]
        first                  = APRILTAG_FAMILIES[self.current_family][0]
        self.current_tag_lbl   = first[0]
        self.current_tag_size  = first[1]
        self.current_pixel_fmt = "Bayer8"
        self.show_dof          = True
        self.show_vertical     = False   # H-FOV by default

        self._params = {}
        for _, k, v in self.CAMERA_FIELDS:
            self._params[k] = float(v)
        for _, k, v in self.DOF_FIELDS:
            self._params[k] = float(v)
        self._params['tag_z']       = "150"
        self._params['tag_lateral'] = "0"
        self._params['tag_angle']   = "0"
        self._params['tag_size']    = self.current_tag_size
        # Default ROI = full sensor
        self._params['roi_w'] = self._params['iw']
        self._params['roi_h'] = self._params['ih']

        self._cone_cache      = ArtistCache()
        self._last_static_key = None
        self._zone_patches    = []
        self._f_use           = None   # set by build_cone_static
        self._fov_rad         = None

        self.fig = plt.figure(figsize=(22, 12))
        self.fig.patch.set_facecolor('#f5f6fa')
        self.fig.canvas.manager.set_window_title(
            'Camera Projection Cone Viewer  v8')

        self._build_layout()
        self._build_section_labels()
        self._build_textboxes()
        self._build_dropdowns()
        self._build_misc_buttons()
        self.update(None)

    # ── Layout ────────────────────────────────────────────────────

    def _build_layout(self):
        # Cone plot: left portion
        self.ax_cone = self.fig.add_axes([0.03, 0.42, 0.54, 0.50])
        # Metrics panel: right portion, full height
        self.ax_metrics = self.fig.add_axes([0.60, 0.02, 0.385, 0.905])
        self.ax_metrics.set_facecolor('#ffffff')
        for sp in self.ax_metrics.spines.values():
            sp.set_edgecolor('#cccccc')

    # ── Section labels ────────────────────────────────────────────

    def _build_section_labels(self):
        ls = dict(fontsize=8, fontweight='bold', color='white',
                  va='center', ha='left')
        sections = [
            # x,     y,     w,     label,                     colour
            (0.030, 0.397, 0.095, '  Dropdowns',              '#37474f'),
            (0.140, 0.397, 0.240, '  Camera / Stream inputs', '#2c3e50'),
            (0.390, 0.397, 0.110, '  DoF & Tag',              '#6a1b9a'),
        ]
        for x0, y0, w, lbl, col in sections:
            bar = self.fig.add_axes([x0, y0, w, 0.018])
            bar.set_facecolor(col)
            bar.axis('off')
            bar.text(0.01, 0.5, lbl, transform=bar.transAxes, **ls)

    # ── Text boxes ────────────────────────────────────────────────

    def _build_textboxes(self):
        self.textboxes = {}
        tb_h     = self.TB_H
        row_step = self.ROW_STEP
        lbl_off  = self.LBL_OFF
        tb_w     = self.TB_W

        # ── Column A: camera/stream — two sub-columns
        #    sub-col 0: indices 0-6  (focal, sensorW/H, W/H, near, far)
        #    sub-col 1: indices 7-12 (fps, roi_x, roi_y, roi_w, roi_h, scale)
        sub_xs = [self.COL_A_X, self.COL_A_X + 0.120]
        top_y  = 0.360
        for i, (lbl, key, default) in enumerate(self.CAMERA_FIELDS):
            sub = 0 if i < 7 else 1
            row = i if i < 7 else i - 7
            x0  = sub_xs[sub]
            y0  = top_y - row * row_step
            # colour: blue for intrinsics/depth, red for stream
            tcol = '#2c3e50' if i < 7 else '#b71c1c'
            bgcol = '#ffffff' if i < 7 else '#ffebee'
            self.fig.text(x0, y0 + lbl_off, lbl,
                          fontsize=6.8, color=tcol, va='bottom')
            ax_tb = self.fig.add_axes([x0, y0, tb_w, tb_h])
            ax_tb.set_facecolor(bgcol)
            tb = TextBox(ax_tb, '', initial=str(float(default)))
            tb.on_submit(lambda val, k=key: self._on_tb_float(k, val))
            self.textboxes[key] = tb

        # ── Column B: DoF then tag position
        col_b_x = self.COL_B_X
        # DoF section
        dof_top = 0.360
        for i, (lbl, key, default) in enumerate(self.DOF_FIELDS):
            y0 = dof_top - i * row_step
            self.fig.text(col_b_x, y0+lbl_off, lbl,
                          fontsize=6.8, color='#00838f', va='bottom')
            ax_tb = self.fig.add_axes([col_b_x, y0, tb_w, tb_h])
            ax_tb.set_facecolor('#e0f7fa')
            tb = TextBox(ax_tb, '', initial=str(float(default)))
            tb.on_submit(lambda val, k=key: self._on_tb_float(k, val))
            self.textboxes[key] = tb

        # Tag section (below DoF with a small gap)
        tag_top = dof_top - len(self.DOF_FIELDS)*row_step - 0.025
        tag_bg = {'tag_z':'#f3e5f5','tag_lateral':'#f3e5f5','tag_angle':'#ede7f6'}
        for i, (lbl, key, default) in enumerate(self.TAG_POS_FIELDS):
            y0 = tag_top - i * row_step
            self.fig.text(col_b_x, y0+lbl_off, lbl,
                          fontsize=6.8, color='#6a1b9a', va='bottom')
            ax_tb = self.fig.add_axes([col_b_x, y0, tb_w, tb_h])
            ax_tb.set_facecolor(tag_bg.get(key,'#f3e5f5'))
            tb = TextBox(ax_tb, '', initial=str(default))
            tb.on_submit(lambda val, k=key: self._on_tb_str(k, val))
            self.textboxes[key] = tb

    def _on_tb_float(self, key, val):
        try: self._params[key] = float(val)
        except ValueError: pass
        self.update(None)

    def _on_tb_str(self, key, val):
        self._params[key] = val
        self.update(None)

    # ── Dropdowns (stacked column) ─────────────────────────────────

    def _build_dropdowns(self):
        lens_names  = list(LENS_TYPES.keys())
        fam_names   = list(APRILTAG_FAMILIES.keys())
        size_labels = [lbl for lbl,_ in APRILTAG_FAMILIES[self.current_family]]
        fmt_names   = list(PIXEL_FORMATS.keys())

        def ph(n): return min(0.021*n + 0.01, 0.40)

        x0  = self.DD_X
        w   = self.DD_W
        h   = self.DD_H
        gap = self.DD_GAP
        y   = self.DD_TOP

        # Each dropdown: label above button, panel opens to the RIGHT
        panel_x = x0 + w + 0.003

        configs = [
            ('Lens',        lens_names,  self.current_lens,      self._select_lens,
             '#e8f0fe', '#1565c0'),
            ('Tag family',  fam_names,   self.current_family,    self._select_family,
             '#fff3e0', '#e65100'),
            ('Tag size',    size_labels, self.current_tag_lbl,   self._select_tag_size,
             '#e8f5e9', '#2e7d32'),
            ('Pixel fmt',   fmt_names,   self.current_pixel_fmt, self._select_pixfmt,
             '#fce4ec', '#880e4f'),
        ]

        self._dd_refs = []
        for label, options, initial, cb, btn_col, act_col in configs:
            self.fig.text(x0, y + h + 0.003, label,
                          fontsize=7, fontweight='bold', color=act_col,
                          va='bottom')
            dd = DropdownMenu(
                self.fig,
                btn_rect   = [x0, y, w, h],
                panel_rect = [panel_x, y, 0.18, ph(len(options))],
                options    = options,
                initial    = initial,
                on_select  = cb,
                btn_color  = btn_col,
                active_color = act_col,
                label_fontsize = 7.5,
                header_color   = act_col,
            )
            self._dd_refs.append(dd)
            y -= gap

        self.dd_lens   = self._dd_refs[0]
        self.dd_family = self._dd_refs[1]
        self.dd_size   = self._dd_refs[2]
        self.dd_pixfmt = self._dd_refs[3]

    def _select_lens(self, label):
        self.current_lens    = label
        f_new                = LENS_TYPES[label]['f_mm']
        self._params['f_mm'] = f_new
        if 'f_mm' in self.textboxes:
            self.textboxes['f_mm'].set_val(str(f_new))
        self._last_static_key = None
        self.update(None)

    def _select_family(self, name):
        self.current_family      = name
        first                    = APRILTAG_FAMILIES[name][0]
        self.current_tag_lbl     = first[0]
        self.current_tag_size    = first[1]
        self._params['tag_size'] = self.current_tag_size
        self.dd_size.set_options([lbl for lbl,_ in APRILTAG_FAMILIES[name]],
                                 new_selection=self.current_tag_lbl)
        self._last_static_key = None
        self.update(None)

    def _select_tag_size(self, label):
        for lbl, sz in APRILTAG_FAMILIES[self.current_family]:
            if lbl == label:
                self.current_tag_lbl     = lbl
                self.current_tag_size    = sz
                self._params['tag_size'] = sz
                break
        self._last_static_key = None
        self.update(None)

    def _select_pixfmt(self, fmt):
        self.current_pixel_fmt = fmt
        self.update(None)

    # ── Misc buttons ──────────────────────────────────────────────

    def _build_misc_buttons(self):
        bs = dict(color='#ecf0f1', hovercolor='#bdc3c7')
        buttons = [
            ([0.140, 0.020, 0.090, 0.024], 'Print K matrix', self._print_k),
            ([0.240, 0.020, 0.085, 0.024], 'Toggle DoF',     self._toggle_dof),
            ([0.335, 0.020, 0.100, 0.024], 'H / V  FOV',     self._toggle_fov),
        ]
        for rect, lbl, cb in buttons:
            ax_b = self.fig.add_axes(rect)
            btn  = Button(ax_b, lbl, **bs)
            btn.label.set_fontsize(8)
            btn.on_clicked(cb)

        self.fig.text(
            0.30, 0.975,
            'Camera Projection Cone Viewer  v8',
            ha='center', va='top', fontsize=11,
            fontweight='bold', color='#2c3e50')

    def _toggle_dof(self, _=None):
        self.show_dof = not self.show_dof
        self.update(None)

    def _toggle_fov(self, _=None):
        self.show_vertical = not self.show_vertical
        self._last_static_key = None   # force full redraw
        self.update(None)

    # ── Helpers ───────────────────────────────────────────────────

    def _calc_dof(self, fx_eff):
        if not self.show_dof: return None, None, None
        p   = self._params
        li  = LENS_TYPES[self.current_lens]
        psz = pixel_size_mm(p['sensor_w'], p['iw'])
        coc = auto_coc(p['sensor_w'], p['sensor_h'])
        return compute_dof(fx_eff*psz, li['f_number'], p['focus_dist'], coc)

    def _parse_tags(self):
        zs     = parse_float_list(self._params.get('tag_z',      ''))
        lats   = parse_float_list(self._params.get('tag_lateral', ''))
        angles = parse_float_list(self._params.get('tag_angle',   ''))
        tag_sz = self._params.get('tag_size', 0.0)
        if not zs: return []
        while len(lats)   < len(zs): lats.append(0.0)
        while len(angles) < len(zs): angles.append(0.0)
        result = []
        for z, lat, ang in zip(zs, lats[:len(zs)], angles[:len(zs)]):
            ac = float(np.clip(ang, -89.9, 89.9))
            result.append((z, lat, ac, effective_tag_size(tag_sz, ac)))
        return result

    def _get_bw(self):
        p = self._params
        return compute_bandwidth(
            p['iw'], p['ih'], p['fps'],
            self.current_pixel_fmt,
            p['roi_x'], p['roi_y'], p['roi_w'], p['roi_h'],
            p['scale'])

    def _print_k(self, _=None):
        p = self._params
        fx, fy, cx, cy = compute_intrinsics(
            p['f_mm'], p['sensor_w'], p['sensor_h'], p['iw'], p['ih'])
        K  = get_K_matrix(fx, fy, cx, cy)
        hf, vf, df = compute_fov(fx, fy, p['iw'], p['ih'])
        bw   = self._get_bw()
        tags = self._parse_tags()
        print("\n── Camera K matrix ──────────────────────────────────")
        print(f"  Lens   : {self.current_lens}")
        print(f"  Family : {self.current_family}  |  Tag: {self.current_tag_lbl}")
        for i, (tz, tlat, ta, te) in enumerate(tags):
            print(f"  Tag {i+1}: z={tz:.0f} mm  lat={tlat:.0f} mm  "
                  f"angle={ta:.1f}°  eff={te:.2f} mm")
        print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
        print(f"  Sensor : {int(p['iw'])}×{int(p['ih'])} px")
        print(f"  H-FOV  : {hf:.2f}°   V-FOV: {vf:.2f}°   Diag: {df:.2f}°")
        print(f"  Format : {bw['pixel_format']}  ({bw['bpp']} B/px)")
        print(f"  Output : {bw['out_w']}×{bw['out_h']} @ {bw['fps']:.1f} fps")
        print(f"  Bytes/frame : {bw['bytes_per_frame']:,.0f}")
        print(f"  Raw    : {bw['raw_mb_s']:.2f} MB/s  "
              f"({bw['raw_mbps']:.1f} Mbps  /  {bw['raw_gbps']:.3f} Gbps)")
        print(f"  MJPEG~ : {bw['comp_mjpeg']:.2f} MB/s")
        print(f"  H.264~ : {bw['comp_h264']:.2f} MB/s")
        print(f"  H.265~ : {bw['comp_h265']:.2f} MB/s")
        print(f"\n  K =\n{K}\n")
        print("──────────────────────────────────────────────────────\n")

    # ── Main update ───────────────────────────────────────────────

    def update(self, _):
        for dd in DropdownMenu._registry:
            if dd._open: dd.close()

        p  = self._params
        fx, fy, cx, cy = compute_intrinsics(
            p['f_mm'], p['sensor_w'], p['sensor_h'], p['iw'], p['ih'])
        iw, ih = p['iw'], p['ih']
        zn     = max(1.0, p['z_near'])
        zf     = max(zn+1.0, p['z_far'])
        tag    = p['tag_size']
        tags   = self._parse_tags()
        t1_eff = tags[0][3] if tags else effective_tag_size(tag, 0.0)

        dof_n, dof_f, hyp = self._calc_dof(fx)
        bw = self._get_bw()

        static_key = (round(fx,3), round(fy,3), round(iw), round(ih),
                      round(zn), round(zf), round(tag,2), self.show_vertical)

        if static_key != self._last_static_key:
            self.ax_cone.cla()
            self._cone_cache.clear_all()
            self._zone_patches, self._f_use, self._fov_rad = build_cone_static(
                self.ax_cone, self._cone_cache,
                fx, fy, iw, ih, zn, zf, tag, self.show_vertical)
            self._last_static_key = static_key

        update_cone_dynamic(
            self.ax_cone, self._cone_cache,
            self._f_use, self._fov_rad, zf,
            dof_n, dof_f, tags, self._zone_patches, self.show_vertical)

        draw_metrics(
            self.ax_metrics,
            fx, fy, cx, cy, iw, ih,
            zn, zf, tag,
            self.current_lens, self.current_family, self.current_tag_lbl,
            tags=tags, bw=bw, show_vertical=self.show_vertical,
            dof_near=dof_n, dof_far=dof_f, hyperfocal=hyp,
            focus_dist=p['focus_dist'] if self.show_dof else None)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    viewer = CameraProjectionViewer()
    viewer.show()
