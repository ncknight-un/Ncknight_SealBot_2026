"""
cube_kinematics.py
==================
Reads tag_paths_ext.csv and tagstamps_ext.csv, computes the velocity,
heading, and orientation (roll/pitch/yaw) of the center of a tagged cube
over time, and writes:

    Full_Data.csv              – all computed fields
    ml_ext.csv                 – time_s, velocity_m_s, roll_deg, pitch_deg, yaw_deg
    velocity_vs_time.png       – velocity curve plot

A constant-velocity Kalman filter is applied to the cube-center position
before velocity is derived, cleanly separating real motion from tag
detection jitter.

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
    --ref-tag        INT    Tag ID used as the world reference (default 0)
    --max-speed      FLOAT  Hard ceiling for plausible speed in m/s (default 3.0)
    --process-noise  FLOAT  Kalman process noise Q (default 0.01 — lower = smoother,
                            higher = follows raw data more closely)
    --meas-noise     FLOAT  Kalman measurement noise R (default 0.1 — higher = smoother)
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
# Kalman Filter (constant-velocity model, 3-axis independent)
# ---------------------------------------------------------------------------

class KalmanFilter3D:
    """
    A simple constant-velocity Kalman filter operating independently on
    each of X, Y, Z.

    State vector per axis: [position, velocity]
    Measurement: [position]

    Args:
        q  – process noise variance (how much we trust the motion model)
             Lower  → smoother, slower to react to real changes
             Higher → noisier, faster to react
        r  – measurement noise variance (how noisy we expect the tag poses to be)
             Higher → smoother, trusts measurements less
    """

    def __init__(self, q: float = 0.01, r: float = 0.1):
        self.q = q
        self.r = r

    def filter(self, times: np.ndarray, positions: np.ndarray) -> tuple:
        """
        Run the filter over a sequence of (time, position) pairs.

        Args:
            times:     1-D array of timestamps in seconds, shape (N,)
            positions: 2-D array of shape (N, 3) — columns are x, y, z

        Returns:
            filtered_pos: (N, 3) smoothed positions
            filtered_vel: (N, 3) smoothed velocities  (m/s per axis)
        """
        N = len(times)
        filtered_pos = np.zeros((N, 3))
        filtered_vel = np.zeros((N, 3))

        for axis in range(3):
            z = positions[:, axis]          # measurements for this axis

            # Initial state: start at first measurement, zero velocity
            x = np.array([z[0], 0.0])       # [pos, vel]
            P = np.eye(2) * 1.0             # initial covariance

            H = np.array([[1.0, 0.0]])      # measurement matrix
            R = np.array([[self.r]])        # measurement noise

            for i in range(N):
                # --- Predict ---
                if i == 0:
                    dt = 0.0
                else:
                    dt = times[i] - times[i - 1]
                    dt = max(dt, 1e-6)      # guard against zero / negative dt

                F = np.array([[1.0, dt],
                              [0.0, 1.0]])  # state transition

                # Process noise matrix (position noise scales with dt²)
                Q = self.q * np.array([
                    [dt**4 / 4, dt**3 / 2],
                    [dt**3 / 2, dt**2    ]
                ])

                x = F @ x
                P = F @ P @ F.T + Q

                # --- Update ---
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)     # Kalman gain
                y = z[i] - (H @ x)[0]              # innovation
                x = x + (K * y).flatten()
                P = (np.eye(2) - K @ H) @ P

                filtered_pos[i, axis] = x[0]
                filtered_vel[i, axis] = x[1]

        return filtered_pos, filtered_vel


# ---------------------------------------------------------------------------
# Tag-pose helpers
# ---------------------------------------------------------------------------

def quaternion_to_rotation(qx, qy, qz, qw):
    return Rotation.from_quat([qx, qy, qz, qw])


def estimate_cube_center(tag_rows):
    if tag_rows.empty:
        return None
    return tag_rows[["tx", "ty", "tz"]].values.mean(axis=0)


def estimate_cube_orientation(tag_rows):
    fwd_tag = tag_rows[tag_rows["tag_id"] == 4]
    if not fwd_tag.empty:
        row = fwd_tag.iloc[0]
        r = quaternion_to_rotation(row.qx, row.qy, row.qz, row.qw)
        return r.as_euler("xyz", degrees=True)

    if tag_rows.empty:
        return np.nan, np.nan, np.nan

    quats = [quaternion_to_rotation(r.qx, r.qy, r.qz, r.qw)
             for _, r in tag_rows.iterrows()]

    if len(quats) == 1:
        return quats[0].as_euler("xyz", degrees=True)

    mean_r = quats[0]
    for q in quats[1:]:
        slerp = Slerp([0, 1], Rotation.concatenate([mean_r, q]))
        mean_r = slerp(0.5)

    return mean_r.as_euler("xyz", degrees=True)


def rotation_from_cube_tags(tag_rows):
    forward_local = np.array([0.0, 0.0, 1.0])
    fwd_tag = tag_rows[tag_rows["tag_id"] == 4]
    if not fwd_tag.empty:
        row = fwd_tag.iloc[0]
        return quaternion_to_rotation(row.qx, row.qy, row.qz, row.qw).as_matrix() @ forward_local
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
            ref_tag_id: int = 0,
            max_speed: float = 3.0,
            process_noise: float = 0.01,
            meas_noise: float = 0.1):

    # --- Load ---------------------------------------------------------------
    tags_df   = pd.read_csv(tags_path,   sep=None, engine="python")
    stamps_df = pd.read_csv(stamps_path, sep=None, engine="python")
    tags_df.columns   = tags_df.columns.str.strip()
    stamps_df.columns = stamps_df.columns.str.strip()

    stamps_df["time_s"] = (stamps_df["timestamp_sec"]
                           + stamps_df["timestamp_ns"] * 1e-9)

    merged = tags_df.merge(stamps_df[["frame_id", "time_s"]], on="frame_id", how="left")
    merged.sort_values(["frame_id", "tag_id"], inplace=True)

    ref_df  = merged[merged["tag_id"] == ref_tag_id].copy()
    cube_df = merged[merged["tag_id"] != ref_tag_id].copy()

    ref_positions = {row["frame_id"]: np.array([row.tx, row.ty, row.tz])
                     for _, row in ref_df.iterrows()}

    # --- Per-frame estimates ------------------------------------------------
    records = []
    for frame_id, group in cube_df.groupby("frame_id", sort=True):
        center = estimate_cube_center(group)
        if center is None:
            continue

        fwd              = rotation_from_cube_tags(group)
        roll, pitch, yaw = estimate_cube_orientation(group)
        ref_pos          = ref_positions.get(frame_id)
        dist_to_ref      = (np.linalg.norm(center - ref_pos)
                            if ref_pos is not None else np.nan)

        records.append({
            "frame_id":      frame_id,
            "time_s":        group["time_s"].iloc[0],
            "cx_raw":        center[0],
            "cy_raw":        center[1],
            "cz_raw":        center[2],
            "roll_deg":      roll,
            "pitch_deg":     pitch,
            "yaw_deg":       yaw,
            "fwd_x":         fwd[0] if fwd is not None else np.nan,
            "fwd_y":         fwd[1] if fwd is not None else np.nan,
            "fwd_z":         fwd[2] if fwd is not None else np.nan,
            "dist_to_ref_m": dist_to_ref,
            "visible_tags":  ",".join(str(t) for t in sorted(group["tag_id"].unique())),
        })

    if not records:
        sys.exit("No cube-tag data found. Check tag IDs and file format.")

    result = pd.DataFrame(records).sort_values("time_s").reset_index(drop=True)

    # --- Remove gross outliers before Kalman (bad tag detections) -----------
    raw_pos = result[["cx_raw", "cy_raw", "cz_raw"]].values
    raw_dt  = np.diff(result["time_s"].values, prepend=result["time_s"].values[0])
    raw_dt[0] = raw_dt[1] if len(raw_dt) > 1 else 1.0
    raw_dt  = np.maximum(raw_dt, 1e-6)
    raw_disp = np.linalg.norm(np.diff(raw_pos, axis=0, prepend=raw_pos[[0]]), axis=1)
    raw_speed = raw_disp / raw_dt

    outlier = raw_speed > max_speed
    n_out = outlier.sum()
    if n_out:
        print(f"  Pre-Kalman outliers interpolated: {n_out} frames (speed > {max_speed} m/s)")
        for col in ["cx_raw", "cy_raw", "cz_raw"]:
            result.loc[outlier, col] = np.nan
        result[["cx_raw", "cy_raw", "cz_raw"]] = (
            result[["cx_raw", "cy_raw", "cz_raw"]]
            .interpolate(method="linear", limit_direction="both")
        )

    # --- Kalman filter on position ------------------------------------------
    print(f"  Running Kalman filter  (Q={process_noise}, R={meas_noise}) ...")
    kf = KalmanFilter3D(q=process_noise, r=meas_noise)
    times    = result["time_s"].values
    raw_pos  = result[["cx_raw", "cy_raw", "cz_raw"]].values

    filt_pos, filt_vel = kf.filter(times, raw_pos)

    result["cx"] = filt_pos[:, 0]
    result["cy"] = filt_pos[:, 1]
    result["cz"] = filt_pos[:, 2]

    # Kalman already gives us per-axis velocity; derive scalar speed & direction
    vx_kf = filt_vel[:, 0]
    vy_kf = filt_vel[:, 1]
    vz_kf = filt_vel[:, 2]

    speed = np.sqrt(vx_kf**2 + vy_kf**2 + vz_kf**2)
    speed[0] = 0.0          # first frame → 0

    vel_mag = speed.copy()
    vel_mag[vel_mag < 1e-9] = 1.0   # avoid divide-by-zero for direction
    result["velocity_m_s"]   = speed
    result["vel_dir_x"]      = vx_kf / vel_mag
    result["vel_dir_y"]      = vy_kf / vel_mag
    result["vel_dir_z"]      = vz_kf / vel_mag
    result["heading_xy_deg"] = np.degrees(np.arctan2(vy_kf, vx_kf))

    # --- Full_Data.csv ------------------------------------------------------
    full_cols = [
        "frame_id", "time_s",
        "cx", "cy", "cz",
        "cx_raw", "cy_raw", "cz_raw",
        "velocity_m_s",
        "vel_dir_x", "vel_dir_y", "vel_dir_z",
        "heading_xy_deg",
        "roll_deg", "pitch_deg", "yaw_deg",
        "fwd_x", "fwd_y", "fwd_z",
        "dist_to_ref_m",
        "visible_tags",
    ]
    result[full_cols].to_csv("Full_Data.csv", index=False, float_format="%.6f")
    print(f"Full_Data.csv written  ({len(result)} frames)")

    # --- ml_ext.csv ---------------------------------------------------------
    result[["time_s", "velocity_m_s", "roll_deg", "pitch_deg", "yaw_deg"]].to_csv(
        "ml_ext.csv", index=False, float_format="%.6f")
    print(f"ml_ext.csv written     ({len(result)} frames)")

    # --- velocity_vs_time.png -----------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))

    # Raw (pre-Kalman) speed for comparison
    raw_disp2 = np.linalg.norm(np.diff(raw_pos, axis=0, prepend=raw_pos[[0]]), axis=1)
    raw_dt2   = np.maximum(np.diff(times, prepend=times[0]), 1e-6)
    raw_spd   = raw_disp2 / raw_dt2
    raw_spd[0] = 0.0
    raw_spd   = np.clip(raw_spd, 0, max_speed * 1.5)   # clip for display only

    ax.plot(times, raw_spd, color="#94A3B8", linewidth=0.7,
            alpha=0.6, label="Raw (pre-Kalman)")
    ax.plot(times, result["velocity_m_s"], color="#2563EB",
            linewidth=1.5, label="Kalman-filtered")
    ax.axhline(result["velocity_m_s"].mean(), color="#DC2626",
               linewidth=0.9, linestyle="--",
               label=f'Mean: {result["velocity_m_s"].mean():.3f} m/s')
    ax.fill_between(times, result["velocity_m_s"], alpha=0.12, color="#2563EB")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Velocity (m/s)", fontsize=11)
    ax.set_title("Cube Center Velocity vs Time  (Kalman-filtered)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("velocity_vs_time.png", dpi=150)
    plt.close(fig)
    print("velocity_vs_time.png written")

    # Summary
    v = result["velocity_m_s"]
    print(f"\n  Velocity (Kalman)  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f} m/s")
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
    parser.add_argument("--tags",          default="tag_paths_ext.csv")
    parser.add_argument("--stamps",        default="tagstamps_ext.csv")
    parser.add_argument("--ref-tag",       type=int,   default=0)
    parser.add_argument("--max-speed",     type=float, default=3.0,
                        help="Outlier ceiling before Kalman (m/s, default 3.0)")
    parser.add_argument("--process-noise", type=float, default=0.01,
                        help="Kalman Q: lower=smoother, higher=follows data (default 0.01)")
    parser.add_argument("--meas-noise",    type=float, default=0.1,
                        help="Kalman R: higher=smoother, lower=trusts data (default 0.1)")
    args = parser.parse_args()

    process(
        tags_path     = args.tags,
        stamps_path   = args.stamps,
        ref_tag_id    = args.ref_tag,
        max_speed     = args.max_speed,
        process_noise = args.process_noise,
        meas_noise    = args.meas_noise,
    )


if __name__ == "__main__":
    main()
