# USED: to convert mp4 video data to images and extract AprilTag poses
# this generates a video output with AprilTag detections and saves tag poses to a CSV file.
# To Run: python videoParser.py

from pathlib import Path
import cv2
import numpy as np
import csv, os
import apriltag
from scipy.spatial.transform import Rotation as R

#######################################################################################
#                                  CONFIG
#######################################################################################

# INPUT MP4 files (change these to your mp4 file paths)
# ext_video_path = "/home/sayantani/external_camera.mp4"
int_video_path = "./test2.mp4"
output_dir = "frames_output"
csv_filename_Ext = "timestamps_ext.csv"
csv_filename_Int = "timestamps_int.csv"
tag_csv_filename_Ext = "tag_paths_ext.csv"
tag_csv_filename_Int = "tag_paths_int.csv"
video_filename_Ext = "output_video_ext.avi"
video_filename_Int = "output_video_int.avi"

os.makedirs(output_dir, exist_ok=True)

############################### External Camera #######################################
# AprilTag detector setup
options = apriltag.DetectorOptions(families='tag36h11') # tag36h11: og  
detector = apriltag.Detector(options)
fx, fy, cx, cy = 600, 600, 320, 240
tag_size = 0.16
K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=np.float64)


############################### Internal Camera #######################################
# AprilTag detector setup
optionsInt = apriltag.DetectorOptions(families='tag16h5') # tag16h5: new
detectorInt = apriltag.Detector(optionsInt)
fxInt, fyInt, cxInt, cyInt = 1806.68775, 1801.14087, 1882.18218, 1404.07528
tag_sizeInt = 0.007725
KInt = np.array([[fxInt, 0, cxInt],
                [0, fyInt, cyInt],
                [0,  0,  1]], dtype=np.float64)

#######################################################################################
#                                  Helper: Open VideoCaps
#######################################################################################

# ext_cap = cv2.VideoCapture(ext_video_path)
int_cap = cv2.VideoCapture(int_video_path)

# if not ext_cap.isOpened():
#     raise RuntimeError(f"Cannot open external video: {ext_video_path}")
if not int_cap.isOpened():
    raise RuntimeError(f"Cannot open internal video: {int_video_path}")

# Retrieve FPS (fallback to 30 if unavailable)
# ext_fps = ext_cap.get(cv2.CAP_PROP_FPS) or 30.0
int_fps = int_cap.get(cv2.CAP_PROP_FPS) or 30.0

#######################################################################################
#                                  Process External Video
#######################################################################################

# video_writer_ext = None

# with open(os.path.join(output_dir, csv_filename_Ext), "w", newline="") as csvfile_Ext, \
#      open(os.path.join(output_dir, tag_csv_filename_Ext), "w", newline="") as tag_csvfile_Ext:

#     writer_int = csv.writer(csvfile_Ext)
#     tag_writer_int = csv.writer(tag_csvfile_Ext)
#     writer_int.writerow(["frame_id", "timestamp_ns", "timestamp_sec"])
#     tag_writer_int.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

#     tag_paths = {}
#     frame_id = 0

#     while True:
#         ret, cv_img = ext_cap.read()
#         if not ret:
#             break

#         # If frame has alpha / 4 channels, convert
#         if cv_img.ndim == 3 and cv_img.shape[2] == 4:
#             cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

#         # Initialize video writer with frame dimensions and FPS
#         if video_writer_ext is None:
#             height, width = cv_img.shape[:2]
#             video_writer_ext = cv2.VideoWriter(
#                 os.path.join(output_dir, video_filename_Ext),
#                 cv2.VideoWriter_fourcc(*'XVID'),
#                 ext_fps,
#                 (width, height)
#             )

#         # Timestamp from frame index and fps
#         timestamp_ns = int((frame_id / ext_fps) * 1e9)
#         timestamp_sec = timestamp_ns / 1e9
#         writer_int.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}"])

#         # AprilTag detection
#         gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#         results = detector.detect(gray)

#         for r in results:
#             tag_id = r.tag_id
#             M, init_error, final_error = detector.detection_pose(r, [fx, fy, cx, cy], tag_size)
#             t = M[:3, 3]
#             R_mat = M[:3, :3]
#             quat = R.from_matrix(R_mat).as_quat()

#             pose = (frame_id, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
#             tag_paths.setdefault(tag_id, []).append(pose)
#             tag_writer_int.writerow([tag_id] + list(pose))

#             # Draw detections
#             (ptA, ptB, ptC, ptD) = r.corners
#             pts = [tuple(map(int, p)) for p in (ptA, ptB, ptC, ptD)]
#             cv2.polylines(cv_img, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
#             cX, cY = map(int, r.center)
#             cv2.circle(cv_img, (cX, cY), 5, (0, 0, 255), -1)
#             cv2.putText(cv_img, f"ID: {tag_id}", (pts[0][0], pts[0][1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # 3D Reprojection
#             dist_coeffs = np.zeros(5)
#             rvec, _ = cv2.Rodrigues(R_mat)
#             tvec = t.reshape(3, 1)

#             origin_3d = np.array([[0, 0, 0]], dtype=float)
#             origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist_coeffs)
#             origin_2d = tuple(np.int32(origin_2d[0, 0]))
#             cv2.circle(cv_img, origin_2d, 6, (0, 255, 255), -1)  # yellow point

#             axis_len = tag_size * 1.5
#             axes_3d = np.float32([
#                 [axis_len, 0, 0],
#                 [0, axis_len, 0],
#                 [0, 0, axis_len]
#             ])
#             axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist_coeffs)
#             axes_2d = axes_2d.reshape(-1, 2).astype(int)

#             cv2.line(cv_img, origin_2d, tuple(axes_2d[0]), (0, 0, 255), 3)
#             cv2.line(cv_img, origin_2d, tuple(axes_2d[1]), (0, 255, 0), 3)
#             cv2.line(cv_img, origin_2d, tuple(axes_2d[2]), (255, 0, 0), 3)

#         # Write the frame to the video
#         video_writer_ext.write(cv_img)
#         frame_id += 1

#     print(f"Extracted {frame_id} frames from External Camera.")

#     # Compute Tag stability for external camera
#     total_frames = frame_id
#     with open(os.path.join(output_dir, 'Tag_stability_ext.csv'), 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['tag_id', 'stability_value'])
#         total_stability = 0
#         static_tag_id = 0

#         # For each frame check if there is any non-static tag detected
#         stability_count = 0
#         for f_id in range(total_frames):
#             tags_in_frame = [tag_id for tag_id, poses in tag_paths.items()
#                                 if tag_id != static_tag_id and any(p[0] == f_id for p in poses)]
#             stability = 1 if tags_in_frame else 0
#             stability_count += stability

#         total_stability = stability_count / total_frames if total_frames > 0 else 0
#         writer.writerow(['Total Stability', total_stability])

#         static_poses = tag_paths.get(static_tag_id, [])
#         static_stability = len(static_poses) / total_frames if total_frames > 0 else 0
#         writer.writerow(['Static Tag Stability', static_stability])

#     print("Stability metrics saved to tag_stability_ext.csv")

#     if video_writer_ext is not None:
#         video_writer_ext.release()

#######################################################################################
#                                  Process Internal Video
#######################################################################################

video_writer_int = None

with open(os.path.join(output_dir, csv_filename_Int), "w", newline="") as csvfile_Int, \
     open(os.path.join(output_dir, tag_csv_filename_Int), "w", newline="") as tag_csvfile_Int:

    writer_ext = csv.writer(csvfile_Int)
    tag_writer_ext = csv.writer(tag_csvfile_Int)
    writer_ext.writerow(["frame_id", "timestamp_ns", "timestamp_sec"])
    tag_writer_ext.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

    tag_paths = {}
    frame_id = 0

    while True:
        ret, cv_img = int_cap.read()
        if not ret:
            break

        if cv_img.ndim == 3 and cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

        if video_writer_int is None:
            height, width = cv_img.shape[:2]
            video_writer_int = cv2.VideoWriter(
                os.path.join(output_dir, video_filename_Int),
                cv2.VideoWriter_fourcc(*'XVID'),
                int_fps,
                (width, height)
            )

        timestamp_ns = int((frame_id / int_fps) * 1e9)
        timestamp_sec = timestamp_ns / 1e9
        writer_ext.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}"])

        # AprilTag detection
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        results = detectorInt.detect(gray)

        for r in results:
            tag_id = r.tag_id
            M, init_error, final_error = detectorInt.detection_pose(r, [fxInt, fyInt, cxInt, cyInt], tag_sizeInt)
            t = M[:3, 3]
            R_mat = M[:3, :3]
            quat = R.from_matrix(R_mat).as_quat()

            pose = (frame_id, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
            tag_paths.setdefault(tag_id, []).append(pose)
            tag_writer_ext.writerow([tag_id] + list(pose))

            # Draw detections
            (ptA, ptB, ptC, ptD) = r.corners
            pts = [tuple(map(int, p)) for p in (ptA, ptB, ptC, ptD)]
            cv2.polylines(cv_img, [np.array(pts, np.int32)], True, (255, 0, 0), 2)
            cX, cY = map(int, r.center)
            cv2.circle(cv_img, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(cv_img, f"ID: {tag_id}", (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 3D Reprojection
            dist_coeffs = np.zeros(5)
            rvec, _ = cv2.Rodrigues(R_mat)
            tvec = t.reshape(3, 1)

            origin_3d = np.array([[0, 0, 0]], dtype=float)
            origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, KInt, dist_coeffs)
            origin_2d = tuple(np.int32(origin_2d[0, 0]))
            cv2.circle(cv_img, origin_2d, 6, (0, 255, 255), -1)

            axis_len = tag_sizeInt * 0.5
            axes_3d = np.float32([
                [axis_len, 0, 0],
                [0, axis_len, 0],
                [0, 0, axis_len]
            ])
            axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, KInt, dist_coeffs)
            axes_2d = axes_2d.reshape(-1, 2).astype(int)

            cv2.line(cv_img, origin_2d, tuple(axes_2d[0]), (0, 0, 255), 3)
            cv2.line(cv_img, origin_2d, tuple(axes_2d[1]), (0, 255, 0), 3)
            cv2.line(cv_img, origin_2d, tuple(axes_2d[2]), (255, 0, 0), 3)

        # Write frame (kept twice in original for unknown reason — now only once)
        video_writer_int.write(cv_img)
        frame_id += 1

    print(f"Extracted {frame_id} frames from Internal Camera.")

    # Compute Tag stability for internal camera
    total_frames = frame_id
    with open(os.path.join(output_dir, 'Tag_stability_int.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tag_id', 'stability_value'])
        total_stability = 0
        for tag_id, poses in tag_paths.items():
            stability = len(poses) / total_frames if total_frames > 0 else 0
            total_stability += stability
            writer.writerow([tag_id, stability])

        avg_stability = total_stability / len(tag_paths) if tag_paths else 0
        writer.writerow(['Average', avg_stability])

    print("Stability metrics saved to tag_stability_int.csv")

    if video_writer_int is not None:
        video_writer_int.release()

# release captures
# ext_cap.release()
int_cap.release()

#######################################################################################
                    # Create Side-by-Side Combined Video
#######################################################################################

# # Combined uses the two output videos we just created
# ext_video_out = os.path.join(output_dir, video_filename_Ext)
# int_video_out = os.path.join(output_dir, video_filename_Int)
# combined_video_path = os.path.join(output_dir, "combined_ext_int.avi")

# ext_cap2 = cv2.VideoCapture(ext_video_out)
# int_cap2 = cv2.VideoCapture(int_video_out)

# fps = ext_cap2.get(cv2.CAP_PROP_FPS) or 30.0

# ext_w = int(ext_cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
# ext_h = int(ext_cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
# int_w = int(int_cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
# int_h = int(int_cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# target_h = max(ext_h, int_h)
# combined_w = ext_w + int_w
# combined_h = target_h

# combined_writer = cv2.VideoWriter(
#     combined_video_path,
#     cv2.VideoWriter_fourcc(*'XVID'),
#     fps,
#     (combined_w, combined_h)
# )

# frame_idx = 0
# while True:
#     # ret_ext, frame_ext = ext_cap2.read()
#     ret_int, frame_int = int_cap2.read()

#     if not ret_ext or not ret_int:
#         break

#     if int_h != target_h:
#         scale = target_h / int_h
#         frame_int = cv2.resize(frame_int, (int(int_w * scale), target_h))
#         frame_ext = cv2.resize(frame_ext, (ext_w, target_h))
#     else:
#         frame_ext = cv2.resize(frame_ext, (ext_w, target_h))

#     combined = np.hstack((frame_ext, frame_int))

#     text = f"Frame Index: {frame_idx}"
#     cv2.putText(
#         combined,
#         text,
#         (20, combined_h - 20),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2,
#         cv2.LINE_AA
#     )

#     combined_writer.write(combined)
#     frame_idx += 1

# print(f"Combined video created: {combined_video_path}")

# ext_cap2.release()
# int_cap2.release()
# combined_writer.release()

#######################################################################################
#                                  Completion Message
#######################################################################################

print(f"Extracted {frame_id} frames and saved video to {os.path.join(output_dir, video_filename_Ext)}")
print(f"Extracted {frame_id} frames and saved video to {os.path.join(output_dir, video_filename_Int)}")

print(f"Timestamps → {csv_filename_Ext}")
print(f"AprilTag poses ext → {tag_csv_filename_Ext}")
print(f"Timestamps → {csv_filename_Int}")
print(f"AprilTag poses int → {tag_csv_filename_Int}")

#######################################################################################
#                                  END OF FILE
#######################################################################################
