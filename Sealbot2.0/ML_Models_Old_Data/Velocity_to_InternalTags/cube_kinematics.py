"""
cube_kinematics.py
==================
Reads tag_paths_ext.csv and tagstamps_ext.csv, computes the velocity,
heading, and orientation (roll/pitch/yaw) of the center of a tagged cube
over time, and writes:

    Full_Data.csv              – all computed fields
    ml_ext.csv                 – time_s, velocity_m_s, roll_deg, pitch_deg, yaw_deg
    velocity_vs_time.png       – velocity curve plot

Tag layout (body frame of the cube):
    Tag 0 – reference (fixed in world; used to measure distance from cube center)
    Tag 1 – left side
    Tag 2 – rear
    Tag 3 – right side
    Tag 4 – forward face
    Tag 5 – top side

Usage:
    python cube_kinematics.py \
        --tags   tag_paths_ext.csv \
        --stamps tagstamps_ext.csv

Optional flags:
    --ref-tag      INT    Tag ID used as the world reference (default 0)
    --smoothing    INT    Rolling-average window for velocity/angles (frames, default 1 = off)
    --max-speed    FLOAT  Hard ceiling for plausible speed in m/s (default 3.0).
                          Frames exceeding this are treated as outliers and interpolated.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quaternion_to_rotation(qx, qy, qz, qw):
    return Rotation.from_quat([qx, qy, qz, qw])


def estimate_cube_center(tag_rows):
    if tag_rows.empty:
        return None
    return tag_rows[["tx", "ty", "tz"]].values.mean(axis=0)


def estimate_cube_orientation(tag_rows):
    """Return (roll_deg, pitch_deg, yaw_deg) from visible face tags."""
    fwd_tag = tag_rows[tag_rows["tag_id"] == 4]
    if not fwd_tag.empty:
        row = fwd_tag.iloc[0]
        r = quaternion_to_rotation(row.qx, row.qy, row.qz, row.qw)
        roll, pitch, yaw = r.as_euler("xyz", degrees=True)
        return roll, pitch, yaw

    if tag_rows.empty:
        return np.nan, np.nan, np.nan

    quats = [quaternion_to_rotation(r.qx, r.qy, r.qz, r.qw)
             for _, r in tag_rows.iterrows()]

    if len(quats) == 1:
        roll, pitch, yaw = quats[0].as_euler("xyz", degrees=True)
        return roll, pitch, yaw

    mean_r = quats[0]
    for q in quats[1:]:
        slerp = Slerp([0, 1], Rotation.concatenate([mean_r, q]))
        mean_r = slerp(0.5)

    roll, pitch, yaw = mean_r.as_euler("xyz", degrees=True)
    return roll, pitch, yaw


def rotation_from_cube_tags(tag_rows):
    forward_local = np.array([0.0, 0.0, 1.0])
    fwd_tag = tag_rows[tag_rows["tag_id"] == 4]
    if not fwd_tag.empty:
        row = fwd_tag.iloc[0]
        R = quaternion_to_rotation(row.qx, row.qy, row.qz, row.qw).as_matrix()
        return R @ forward_local
    if tag_rows.empty:
        return None
    vecs = [quaternion_to_rotation(r.qx, r.qy, r.qz, r.qw).as_matrix() @ forward_local
            for _, r in tag_rows.iterrows()]
    mean_fwd = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_fwd)
    return mean_fwd / norm if norm > 1e-9 else None


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(tags_path: str, stamps_path: str,
            ref_tag_id: int = 0, smoothing: int = 1,
            max_speed: float = 3.0):

    # --- Load data ----------------------------------------------------------
    tags_df  = pd.read_csv(tags_path,  sep=None, engine="python")
    stamps_df = pd.read_csv(stamps_path, sep=None, engine="python")
    tags_df.columns  = tags_df.columns.str.strip()
    stamps_df.columns = stamps_df.columns.str.strip()

    for col_set, fname in [
        ({"tag_id","frame_id","tx","ty","tz","qx","qy","qz","qw"}, tags_path),
        ({"frame_id","timestamp_ns","timestamp_sec"}, stamps_path),
    ]:
        missing = col_set - set(pd.read_csv(fname, nrows=0, sep=None, engine="python")
                                  .columns.str.strip())
        if missing:
            sys.exit(f"{fname} is missing columns: {missing}")

    stamps_df["time_s"] = (stamps_df["timestamp_sec"]
                           + stamps_df["timestamp_ns"] * 1e-9)

    merged = tags_df.merge(stamps_df[["frame_id","time_s"]], on="frame_id", how="left")
    merged.sort_values(["frame_id","tag_id"], inplace=True)

    ref_df   = merged[merged["tag_id"] == ref_tag_id].copy()
    cube_df  = merged[merged["tag_id"] != ref_tag_id].copy()

    ref_positions = {row["frame_id"]: np.array([row.tx, row.ty, row.tz])
                     for _, row in ref_df.iterrows()}

    # --- Per-frame cube centre & orientation --------------------------------
    records = []
    for frame_id, group in cube_df.groupby("frame_id", sort=True):
        time_s = group["time_s"].iloc[0]
        center = estimate_cube_center(group)
        if center is None:
            continue

        fwd               = rotation_from_cube_tags(group)
        roll, pitch, yaw  = estimate_cube_orientation(group)
        ref_pos           = ref_positions.get(frame_id)
        dist_to_ref       = (np.linalg.norm(center - ref_pos)
                             if ref_pos is not None else np.nan)

        records.append({
            "frame_id":     frame_id,
            "time_s":       time_s,
            "cx":           center[0],
            "cy":           center[1],
            "cz":           center[2],
            "roll_deg":     roll,
            "pitch_deg":    pitch,
            "yaw_deg":      yaw,
            "fwd_x":        fwd[0] if fwd is not None else np.nan,
            "fwd_y":        fwd[1] if fwd is not None else np.nan,
            "fwd_z":        fwd[2] if fwd is not None else np.nan,
            "dist_to_ref_m": dist_to_ref,
            "visible_tags": ",".join(str(t) for t in sorted(group["tag_id"].unique())),
        })

    if not records:
        sys.exit("No cube-tag data found. Check tag IDs and file format.")

    result = pd.DataFrame(records).sort_values("time_s").reset_index(drop=True)

    # --- Velocity (raw) -----------------------------------------------------
    dt = result["time_s"].diff()
    dx = result["cx"].diff()
    dy = result["cy"].diff()
    dz = result["cz"].diff()

    raw_speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt

    # --- Outlier removal ----------------------------------------------------
    # Flag frames where raw speed exceeds max_speed as outliers; also catch
    # near-zero or negative dt (duplicate / out-of-order timestamps).
    bad_dt    = dt <= 1e-6
    bad_speed = raw_speed > max_speed

    outlier = bad_dt | bad_speed
    n_outliers = outlier.sum()
    if n_outliers:
        print(f"  Outlier frames removed/interpolated: {n_outliers} "
              f"(speed > {max_speed} m/s or bad dt)")

    # Replace outlier positions with NaN then linearly interpolate so velocity
    # is recalculated cleanly.
    for col in ["cx", "cy", "cz"]:
        result.loc[outlier, col] = np.nan
    result[["cx","cy","cz"]] = result[["cx","cy","cz"]].interpolate(
        method="linear", limit_direction="both")

    # Recompute velocity on cleaned positions
    dt    = result["time_s"].diff()
    dx    = result["cx"].diff()
    dy    = result["cy"].diff()
    dz    = result["cz"].diff()
    speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt

    # First frame → 0 instead of NaN
    speed.iloc[0] = 0.0

    vel_mag = np.sqrt(dx**2 + dy**2 + dz**2)
    vx = (dx / vel_mag).where(vel_mag > 0, 0)
    vy = (dy / vel_mag).where(vel_mag > 0, 0)
    vz = (dz / vel_mag).where(vel_mag > 0, 0)

    heading_deg = np.degrees(np.arctan2(dy, dx))

    result["velocity_m_s"]   = speed
    result["vel_dir_x"]      = vx
    result["vel_dir_y"]      = vy
    result["vel_dir_z"]      = vz
    result["heading_xy_deg"] = heading_deg

    # Optional rolling smoothing
    if smoothing > 1:
        for col in ["velocity_m_s","vel_dir_x","vel_dir_y","vel_dir_z",
                    "heading_xy_deg","roll_deg","pitch_deg","yaw_deg"]:
            result[col] = (result[col]
                           .rolling(window=smoothing, center=True, min_periods=1)
                           .mean())

    # --- Full_Data.csv ------------------------------------------------------
    full_cols = [
        "frame_id","time_s",
        "cx","cy","cz",
        "velocity_m_s",
        "vel_dir_x","vel_dir_y","vel_dir_z",
        "heading_xy_deg",
        "roll_deg","pitch_deg","yaw_deg",
        "fwd_x","fwd_y","fwd_z",
        "dist_to_ref_m",
        "visible_tags",
    ]
    result[full_cols].to_csv("Full_Data.csv", index=False, float_format="%.6f")
    print(f"Full_Data.csv written  ({len(result)} frames)")

    # --- ml_ext.csv ---------------------------------------------------------
    ml = result[["time_s","velocity_m_s","roll_deg","pitch_deg","yaw_deg"]].copy()
    ml.to_csv("ml_ext.csv", index=False, float_format="%.6f")
    print(f"ml_ext.csv written     ({len(ml)} frames)")

    # --- velocity_vs_time.png -----------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result["time_s"], result["velocity_m_s"],
            color="#2563EB", linewidth=1.2, label="Velocity")
    ax.axhline(result["velocity_m_s"].mean(), color="#DC2626",
               linewidth=0.9, linestyle="--",
               label=f'Mean: {result["velocity_m_s"].mean():.3f} m/s')
    ax.fill_between(result["time_s"], result["velocity_m_s"],
                    alpha=0.12, color="#2563EB")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Velocity (m/s)", fontsize=11)
    ax.set_title("Cube Center Velocity vs Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("velocity_vs_time.png", dpi=150)
    plt.close(fig)
    print("velocity_vs_time.png written")

    # Summary
    v = result["velocity_m_s"].dropna()
    print(f"\n  Velocity   min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f} m/s")
    dist = result["dist_to_ref_m"].dropna()
    if not dist.empty:
        print(f"  Dist-to-ref  min={dist.min():.4f}  max={dist.max():.4f}"
              f"  mean={dist.mean():.4f} m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute cube-center velocity & orientation from AprilTag poses."
    )
    parser.add_argument("--tags",      default="tag_paths_ext.csv")
    parser.add_argument("--stamps",    default="tagstamps_ext.csv")
    parser.add_argument("--ref-tag",   type=int,   default=0)
    parser.add_argument("--smoothing", type=int,   default=1,
                        help="Rolling-average window (default 1 = off)")
    parser.add_argument("--max-speed", type=float, default=3.0,
                        help="Plausible speed ceiling in m/s for outlier removal (default 3.0)")
    args = parser.parse_args()

    process(
        tags_path  = args.tags,
        stamps_path= args.stamps,
        ref_tag_id = args.ref_tag,
        smoothing  = args.smoothing,
        max_speed  = args.max_speed,
    )


if __name__ == "__main__":
    main()
