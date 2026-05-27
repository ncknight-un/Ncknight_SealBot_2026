"""
build_clean_data.py
===================
Merges internal tag pose data with ML_ext.csv (robot velocity + orientation)
to produce clean_data.csv for ML training.

All orientations are stored as unit quaternions (qx, qy, qz, qw) with
consistent hemisphere normalisation (qw >= 0) to avoid the double-cover
ambiguity. This avoids gimbal lock and is more compact than sin/cos pairs.

Inputs:
    tag_paths_int.csv   - tag_id  frame_id  tx ty tz  qx qy qz qw
    timestamps_int.csv  - frame_id  timestamp_ns  timestamp_sec
    ml_ext.csv          - time_s  velocity_m_s  roll_deg  pitch_deg  yaw_deg

Output:
    clean_data.csv

Column layout (designed for clean PyTorch slicing):
    [frame_id, time_s]                            - identifiers, exclude from model

    POSE INPUTS  (scale x/y/z only — quaternion components are already [-1,1]):
      tag{N}_x, tag{N}_y, tag{N}_z               - position relative to tag 19
      tag{N}_qx, tag{N}_qy, tag{N}_qz, tag{N}_qw - orientation relative to tag 19
                                                    18 tags × 7 = 126 cols
      left_disp_avg, left_visible_count           - left  side aggregates
      right_disp_avg, right_visible_count         - right side aggregates
    Total pose input cols: 126 + 4 = 130

    FLAG INPUTS  (do NOT scale - binary 0/1):
      tag{N}_missing                              - 1 if not visible  (18 cols)

    Total inputs: 148

    OUTPUTS (5 values):
      velocity_m_s
      out_qx, out_qy, out_qz, out_qw             - vehicle heading as quaternion

    NOTE: to recover Euler angles from output quaternion:
        from scipy.spatial.transform import Rotation
        roll, pitch, yaw = Rotation.from_quat([qx,qy,qz,qw]).as_euler('xyz', degrees=True)

Usage:
    python build_clean_data.py \
        --int-tags   tag_paths_int.csv \
        --int-stamps timestamps_int.csv \
        --ml         ml_ext.csv \
        --output     clean_data.csv

Optional:
    --ref-tag    INT  Stationary reference tag ID (default 19)
    --left-tags  comma-separated tag IDs (default 0,1,2,3,4,5,6,7,8)
    --right-tags comma-separated tag IDs (default 9,10,11,12,13,14,15,16,17)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_quat(qx, qy, qz, qw):
    """
    Normalise to unit length and flip to consistent hemisphere (qw >= 0).
    This resolves the quaternion double-cover ambiguity so the model sees
    a consistent representation of the same rotation.
    """
    q = np.array([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    if q[3] < 0:      # qw < 0 → flip all components
        q = -q
    return q[0], q[1], q[2], q[3]


def relative_pose_quat(tx, ty, tz, qx, qy, qz, qw,
                       ref_tx, ref_ty, ref_tz, ref_qx, ref_qy, ref_qz, ref_qw):
    """
    Express a tag's pose in the reference tag's local frame.

    Position : p_rel = R_ref^T · (p_tag - p_ref)
    Rotation : R_rel = R_ref^T · R_tag  → returned as normalised quaternion

    Returns (rx, ry, rz, rqx, rqy, rqz, rqw)
    """
    R_ref = Rotation.from_quat([ref_qx, ref_qy, ref_qz, ref_qw])
    R_tag = Rotation.from_quat([qx, qy, qz, qw])

    p_diff = np.array([tx - ref_tx, ty - ref_ty, tz - ref_tz])
    p_rel  = R_ref.inv().apply(p_diff)

    R_rel = R_ref.inv() * R_tag
    rqx, rqy, rqz, rqw = R_rel.as_quat()          # scipy returns [x,y,z,w]
    rqx, rqy, rqz, rqw = normalise_quat(rqx, rqy, rqz, rqw)

    return p_rel[0], p_rel[1], p_rel[2], rqx, rqy, rqz, rqw


def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles (degrees, XYZ) to a normalised quaternion."""
    r = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True)
    qx, qy, qz, qw = r.as_quat()
    return normalise_quat(qx, qy, qz, qw)


def displacement_magnitude(tx, ty, tz, ref_tx, ref_ty, ref_tz):
    return float(np.linalg.norm([tx - ref_tx, ty - ref_ty, tz - ref_tz]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(int_tags_path, int_stamps_path, ml_path, output_path,
            ref_tag_id=19, left_tags=None, right_tags=None):

    if left_tags  is None: left_tags  = list(range(0, 9))
    if right_tags is None: right_tags = list(range(9, 18))
    all_mobile_tags = sorted(set(left_tags + right_tags))

    # --- Load ---------------------------------------------------------------
    tags_df   = pd.read_csv(int_tags_path,   sep=None, engine="python")
    stamps_df = pd.read_csv(int_stamps_path, sep=None, engine="python")
    ml_df     = pd.read_csv(ml_path,         sep=None, engine="python")

    tags_df.columns   = tags_df.columns.str.strip()
    stamps_df.columns = stamps_df.columns.str.strip()
    ml_df.columns     = ml_df.columns.str.strip()

    for df, name, required in [
        (tags_df,   int_tags_path,   {"tag_id","frame_id","tx","ty","tz","qx","qy","qz","qw"}),
        (stamps_df, int_stamps_path, {"frame_id","timestamp_ns","timestamp_sec"}),
        (ml_df,     ml_path,         {"time_s","velocity_m_s","roll_deg","pitch_deg","yaw_deg"}),
    ]:
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"{name} is missing columns: {missing}")

    stamps_df["time_s"] = (stamps_df["timestamp_sec"]
                           + stamps_df["timestamp_ns"] * 1e-9)

    merged = tags_df.merge(stamps_df[["frame_id","time_s"]], on="frame_id", how="left")
    merged.sort_values(["frame_id","tag_id"], inplace=True)

    ml_df = ml_df.sort_values("time_s").reset_index(drop=True)

    # Convert ml_ext Euler angles to quaternion outputs
    qs = [euler_to_quat(row.roll_deg, row.pitch_deg, row.yaw_deg)
          for _, row in ml_df.iterrows()]
    ml_df["out_qx"] = [q[0] for q in qs]
    ml_df["out_qy"] = [q[1] for q in qs]
    ml_df["out_qz"] = [q[2] for q in qs]
    ml_df["out_qw"] = [q[3] for q in qs]

    # --- Build one row per frame --------------------------------------------
    rows = []

    for frame_id, frame in merged.groupby("frame_id", sort=True):
        time_s = frame["time_s"].iloc[0]
        row = {"frame_id": frame_id, "time_s": time_s}

        ref_row = frame[frame["tag_id"] == ref_tag_id]
        has_ref = not ref_row.empty

        if has_ref:
            ref = ref_row.iloc[0]
            ref_tx, ref_ty, ref_tz = ref.tx, ref.ty, ref.tz
            ref_qx, ref_qy, ref_qz, ref_qw = ref.qx, ref.qy, ref.qz, ref.qw
        else:
            ref_tx = ref_ty = ref_tz = 0.0
            ref_qx = ref_qy = ref_qz = 0.0
            ref_qw = 1.0

        left_disps  = []
        right_disps = []
        left_vis    = 0
        right_vis   = 0

        for tid in all_mobile_tags:
            tag_row = frame[frame["tag_id"] == tid]
            visible = (not tag_row.empty) and has_ref

            if visible:
                t = tag_row.iloc[0]
                rx, ry, rz, rqx, rqy, rqz, rqw = relative_pose_quat(
                    t.tx, t.ty, t.tz, t.qx, t.qy, t.qz, t.qw,
                    ref_tx, ref_ty, ref_tz, ref_qx, ref_qy, ref_qz, ref_qw
                )
                disp         = displacement_magnitude(t.tx, t.ty, t.tz, ref_tx, ref_ty, ref_tz)
                missing_flag = 0
            else:
                # Identity quaternion (0,0,0,1) = no rotation, zeroed position
                rx = ry = rz = 0.0
                rqx = rqy = rqz = 0.0
                rqw = 1.0
                disp         = 0.0
                missing_flag = 1

            row[f"tag{tid}_x"]   = rx
            row[f"tag{tid}_y"]   = ry
            row[f"tag{tid}_z"]   = rz
            row[f"tag{tid}_qx"]  = rqx
            row[f"tag{tid}_qy"]  = rqy
            row[f"tag{tid}_qz"]  = rqz
            row[f"tag{tid}_qw"]  = rqw
            row[f"tag{tid}_missing"] = missing_flag

            if tid in left_tags and visible:
                left_disps.append(disp)
                left_vis += 1
            if tid in right_tags and visible:
                right_disps.append(disp)
                right_vis += 1

        row["left_disp_avg"]       = float(np.mean(left_disps))  if left_disps  else 0.0
        row["left_visible_count"]  = left_vis
        row["right_disp_avg"]      = float(np.mean(right_disps)) if right_disps else 0.0
        row["right_visible_count"] = right_vis

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)

    # --- Timestamp-tolerant inner join with ml_ext.csv ----------------------
    MAX_GAP_S = 0.020   # 20 ms

    result_sorted = result.sort_values("time_s").reset_index(drop=True)
    ml_sorted     = ml_df.sort_values("time_s").reset_index(drop=True)

    ml_out_cols = ["time_s", "velocity_m_s", "out_qx", "out_qy", "out_qz", "out_qw"]

    merged_ml = pd.merge_asof(
        result_sorted,
        ml_sorted[ml_out_cols],
        on="time_s",
        direction="nearest",
        suffixes=("", "_ml")
    )

    merged_ml["_ml_time"] = pd.merge_asof(
        result_sorted[["time_s"]],
        ml_sorted[["time_s"]].rename(columns={"time_s": "_ml_t"}),
        left_on="time_s", right_on="_ml_t",
        direction="nearest"
    )["_ml_t"].values

    merged_ml["_gap"] = (merged_ml["time_s"] - merged_ml["_ml_time"]).abs()
    n_before  = len(merged_ml)
    merged_ml = merged_ml[merged_ml["_gap"] <= MAX_GAP_S].reset_index(drop=True)
    merged_ml = merged_ml.drop(columns=["_ml_time", "_gap"])
    n_dropped = n_before - len(merged_ml)
    if n_dropped:
        print(f"  Frames dropped (gap > {MAX_GAP_S*1000:.0f}ms): {n_dropped}")

    # --- Column order -------------------------------------------------------
    pose_input_cols = []
    for tid in all_mobile_tags:
        pose_input_cols += [
            f"tag{tid}_x",  f"tag{tid}_y",  f"tag{tid}_z",
            f"tag{tid}_qx", f"tag{tid}_qy", f"tag{tid}_qz", f"tag{tid}_qw",
        ]
    pose_input_cols += ["left_disp_avg",  "left_visible_count",
                        "right_disp_avg", "right_visible_count"]

    flag_cols   = [f"tag{tid}_missing" for tid in all_mobile_tags]
    output_cols = ["velocity_m_s", "out_qx", "out_qy", "out_qz", "out_qw"]

    final_cols = (
        ["frame_id", "time_s"]
        + pose_input_cols
        + flag_cols
        + output_cols
    )

    merged_ml[final_cols].to_csv(output_path, index=False, float_format="%.6f")

    # --- Summary ------------------------------------------------------------
    n       = len(merged_ml)
    n_pose  = len(pose_input_cols)
    n_flags = len(flag_cols)
    n_out   = len(output_cols)

    print(f"clean_data.csv written  ({n} frames)")
    print(f"  Reference tag         : {ref_tag_id}")
    print(f"  Left  tags            : {left_tags}")
    print(f"  Right tags            : {right_tags}")
    print(f"\n  Column layout for PyTorch:")
    print(f"    Pose inputs  : cols 0   to {n_pose - 1}  ({n_pose} cols)")
    print(f"                   scale x/y/z only — quat components already in [-1, 1]")
    print(f"    Flag inputs  : cols {n_pose} to {n_pose + n_flags - 1}  ({n_flags} cols) — do NOT scale")
    print(f"    Total inputs : {n_pose + n_flags}")
    print(f"    Outputs      : {output_cols}  ({n_out} cols)")
    print(f"    NOTE         : recover Euler angles via")
    print(f"                   Rotation.from_quat([qx,qy,qz,qw]).as_euler('xyz', degrees=True)")
    print()
    for tid in all_mobile_tags:
        vis = n - int(merged_ml[f"tag{tid}_missing"].sum())
        pct = (100 * vis / n) if n > 0 else 0.0
        print(f"  Tag {tid:2d}  visible {vis}/{n} frames  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build clean_data.csv from internal tags + ML_ext.csv"
    )
    parser.add_argument("--int-tags",   default="tag_paths_int.csv")
    parser.add_argument("--int-stamps", default="timestamps_int.csv")
    parser.add_argument("--ml",         default="ml_ext.csv")
    parser.add_argument("--output",     default="clean_data.csv")
    parser.add_argument("--ref-tag",    type=int, default=19)
    parser.add_argument("--left-tags",  default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--right-tags", default="9,10,11,12,13,14,15,16,17")
    args = parser.parse_args()

    process(
        int_tags_path   = args.int_tags,
        int_stamps_path = args.int_stamps,
        ml_path         = args.ml,
        output_path     = args.output,
        ref_tag_id      = args.ref_tag,
        left_tags       = [int(x) for x in args.left_tags.split(",")],
        right_tags      = [int(x) for x in args.right_tags.split(",")],
    )


if __name__ == "__main__":
    main()
