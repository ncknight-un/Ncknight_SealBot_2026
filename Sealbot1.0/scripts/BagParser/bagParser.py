# USED: to convert rosbag2 data to images and extract AprilTag poses
# this generates a video output with AprilTag detections and saves tag poses to a CSV file.
# To Run:   pyenv shell realsense-env  || python bagParser.py

from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2
import numpy as np
import csv, os
import dt_apriltags as apriltag
from scipy.spatial.transform import Rotation as R

#######################################################################################
#                                  CONFIG
#######################################################################################

bag_path = "/home/nolan-knight/FinalProject/ros2_ws/src/Prior_Data/rosbag2_2025_11_25-21_28_00"
output_dir = "frames_output"
csv_filename_Ext = "timestamps_ext.csv"
csv_filename_Int = "timestamps_int.csv"
tag_csv_filename_Ext = "tag_paths_ext.csv"
tag_csv_filename_Int = "tag_paths_int.csv"
video_filename_Ext = "output_video_ext.avi"
video_filename_Int = "output_video_int.avi"

os.makedirs(output_dir, exist_ok=True)

############################### External Camera #######################################
topic_name = "/camera/camera/color/image_raw"   
# AprilTag detector setup
detector = apriltag.Detector(families='tag36h11')
fx, fy, cx, cy = 600, 600, 320, 240
tag_size = 0.16
K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=np.float64)


############################### Internal Camera #######################################
topic_nameInt = "/camera/image"   
# AprilTag detector setup
detector = apriltag.Detector(families='tag36h11')
fxInt, fyInt, cxInt, cyInt = 1806.68775, 1801.14087, 1882.18218, 1404.07528   # With OpenCV Calib Matrix. TODO: verify std units here same as openCV.
tag_sizeInt = 0.007725     # The black square width ≈ 0.75 × 10.3 mm = 7.725 mm.   || 10.3 mm is with border
KInt = np.array([[fxInt, 0, cxInt],
                [0, fyInt, cyInt],
                [0,  0,  1]], dtype=np.float64)

#######################################################################################
#                                  Helper Function
#######################################################################################

def message_to_cvimage(msg):
    """Convert sensor_msgs/msg/Image to OpenCV image (NumPy array)."""
    dtype_map = {
        'rgb8': np.uint8,
        'bgr8': np.uint8,
        'mono8': np.uint8,
    }

    if msg.encoding not in dtype_map:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    img = np.frombuffer(msg.data, dtype=dtype_map[msg.encoding]).reshape(msg.height, msg.width, -1)

    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

#######################################################################################
#                                  Read ROSBAG
#######################################################################################

with AnyReader([Path(bag_path)]) as reader:
    # Optional: inspect topics
    print("Topics found:", [x.topic for x in reader.connections])

    # Find the desired topic connection
    conn = next(c for c in reader.connections if c.topic == topic_name)
    connInt = next(c for c in reader.connections if c.topic == topic_nameInt)


    # Initialize video writers separately for external and internal cameras
    video_writer_ext = None
    video_writer_int = None

    with open(os.path.join(output_dir, csv_filename_Ext), "w", newline="") as csvfile_Ext, \
         open(os.path.join(output_dir, tag_csv_filename_Ext), "w", newline="") as tag_csvfile_Ext, \
            open(os.path.join(output_dir, csv_filename_Int), "w", newline="") as csvfile_Int, \
            open(os.path.join(output_dir, tag_csv_filename_Int), "w", newline="") as tag_csvfile_Int:

        ######################################################################################
        # External Tag Detection Loop

        writer_int = csv.writer(csvfile_Ext)
        tag_writer_int = csv.writer(tag_csvfile_Ext)
        writer_int.writerow(["frame_id", "timestamp_ns", "timestamp_sec"])
        tag_writer_int.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        tag_paths = {}
        frame_id = 0

        for connection, timestamp, rawdata in reader.messages(connections=[conn]):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Convert ROS Image → OpenCV
            cv_img = message_to_cvimage(msg).copy()
            if cv_img.ndim == 3 and cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

            # Initialize video writer with frame dimensions and FPS
            if video_writer_ext is None:
                height, width = cv_img.shape[:2]
                video_writer_ext = cv2.VideoWriter(
                    os.path.join(output_dir, video_filename_Ext),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    30,  # Assuming 30 FPS
                    (width, height)
                )

            timestamp_ns = timestamp
            timestamp_sec = timestamp_ns / 1e9
            writer_int.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}"])

            # AprilTag detection
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=tag_size)

            for r in results:
                tag_id = r.tag_id
                t = r.pose_t.flatten()
                R_mat = r.pose_R
                quat = R.from_matrix(R_mat).as_quat()

                pose = (frame_id, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
                tag_paths.setdefault(tag_id, []).append(pose)
                tag_writer_int.writerow([tag_id] + list(pose))

                # Draw detections
                (ptA, ptB, ptC, ptD) = r.corners
                pts = [tuple(map(int, p)) for p in (ptA, ptB, ptC, ptD)]
                cv2.polylines(cv_img, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
                cX, cY = map(int, r.center)
                cv2.circle(cv_img, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(cv_img, f"ID: {tag_id}", (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               
                # 3D Reprojection of the detected tag pose.
                
                dist_coeffs = np.zeros(5)
                # Rodrigues rotation
                rvec, _ = cv2.Rodrigues(R_mat)
                tvec = t.reshape(3, 1)

                # --- 1) Project tag origin ---
                origin_3d = np.array([[0, 0, 0]], dtype=float)
                origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist_coeffs)
                origin_2d = tuple(np.int32(origin_2d[0, 0]))

                cv2.circle(cv_img, origin_2d, 6, (0, 255, 255), -1)  # yellow point

                # --- 2) Project axis endpoints ---
                axis_len = tag_size * 1.5

                axes_3d = np.float32([
                    [axis_len, 0, 0],   # X → red
                    [0, axis_len, 0],   # Y → green
                    [0, 0, axis_len]    # Z → blue
                ])

                axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist_coeffs)
                axes_2d = axes_2d.reshape(-1, 2).astype(int)

                cv2.line(cv_img, origin_2d, tuple(axes_2d[0]), (0, 0, 255), 3)   # X = red
                cv2.line(cv_img, origin_2d, tuple(axes_2d[1]), (0, 255, 0), 3)   # Y = green
                cv2.line(cv_img, origin_2d, tuple(axes_2d[2]), (255, 0, 0), 3)   # Z = blue

            # Overlay frame index
            text = f"External Frame Index: {frame_id}"
            cv2.putText(
                cv_img,
                text,
                (20, height - 20),   # bottom-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),           # yellow for visibility
                2,
                cv2.LINE_AA
            )

            # Write the frame to the video
            video_writer_ext.write(cv_img)

            frame_id += 1
        print(f"Extracted {frame_id} frames from External Camera.")

        # Compute Tag stability: Tag stability means different here, not all tags would be visible at all frames,
        # but at least one should be at all times along with the static one. 

        total_frames = frame_id 
        with open('frames_output/Tag_stability_ext.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['tag_id', 'stability_value'])
            total_stability = 0
            static_tag_id = 0  # Assuming tag ID 0 is static and always present.

            # Caculate if every frame has at least one tag detected : do this for all tag id except the static tad_id
            for f_id in range(total_frames):
                tags_in_frame = [tag_id for tag_id, poses in tag_paths.items() 
                                 if tag_id != static_tag_id and any(p[0] == f_id for p in poses)]

                stability = 1 if tags_in_frame else 0

            total_stability += stability / total_frames  # Normalize by total frames
            writer.writerow(['Total Stability', total_stability])

            # For static tag:
            static_poses = tag_paths.get(static_tag_id, [])
            static_stability = len(static_poses) / total_frames if total_frames > 0 else 0
            writer.writerow(['Static Tag Stability', static_stability])
        
        print("Stability metrics saved to tag_stability_ext.csv")  

        ######################################################################################
        # Internal Tag Detection Loop
        
        writer_ext = csv.writer(csvfile_Int)
        tag_writer_ext = csv.writer(tag_csvfile_Int)
        writer_ext.writerow(["frame_id", "timestamp_ns", "timestamp_sec"])
        tag_writer_ext.writerow(['tag_id', 'frame_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        tag_paths = {}
        frame_id = 0      
        
        for connection, timestamp, rawdata in reader.messages(connections=[connInt]):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Convert ROS Image → OpenCV
            cv_img = message_to_cvimage(msg).copy()
            if cv_img.ndim == 3 and cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
            
            if video_writer_int is None:
                height, width = cv_img.shape[:2]
                video_writer_int = cv2.VideoWriter(
                    os.path.join(output_dir, video_filename_Int),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    30,  # Assuming 30 FPS
                    (width, height)
                )

            timestamp_ns = timestamp
            timestamp_sec = timestamp_ns / 1e9
            writer_ext.writerow([frame_id, int(timestamp_ns), f"{timestamp_sec:.6f}"])

            # AprilTag detection
            if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_img  # already grayscale

            # Sharpen Image before detecting
            kernel = np.array([[0, -1,  0],
                            [-1, 5, -1],
                            [0, -1,  0]], dtype=np.float32)
            gray_sharp = cv2.filter2D(gray, -1, kernel)

            # Also apply CLAHE to fix the contrast/exposure unevenness
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_sharp = clahe.apply(gray_sharp)

            results = detector.detect(gray_sharp, estimate_tag_pose=True, camera_params=[fxInt, fyInt, cxInt, cyInt], tag_size=tag_sizeInt)

            for r in results:
                tag_id = r.tag_id
                t = r.pose_t.flatten()
                R_mat = r.pose_R
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 3D Reprojection of the detected tag pose.
                dist_coeffs = np.zeros(5)

                # Rodrigues rotation
                rvec, _ = cv2.Rodrigues(R_mat)
                tvec = t.reshape(3, 1)

                # --- 1) Project tag origin ---
                origin_3d = np.array([[0, 0, 0]], dtype=float)
                origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, KInt, dist_coeffs)
                origin_2d = tuple(np.int32(origin_2d[0, 0]))

                cv2.circle(cv_img, origin_2d, 6, (0, 255, 255), -1)  # yellow point

                # --- 2) Project axis endpoints ---
                axis_len = tag_sizeInt * 0.5

                axes_3d = np.float32([
                    [axis_len, 0, 0],   # X → red
                    [0, axis_len, 0],   # Y → green
                    [0, 0, axis_len]    # Z → blue
                ])

                axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, KInt, dist_coeffs)
                axes_2d = axes_2d.reshape(-1, 2).astype(int)

                cv2.line(cv_img, origin_2d, tuple(axes_2d[0]), (0, 0, 255), 3)  
                cv2.line(cv_img, origin_2d, tuple(axes_2d[1]), (0, 255, 0), 3)  
                cv2.line(cv_img, origin_2d, tuple(axes_2d[2]), (255, 0, 0), 3) 

            # Overlay frame index
            text = f"Internal Frame Index: {frame_id}"
            cv2.putText(
                cv_img,
                text,
                (20, height - 20),   # bottom-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),           # yellow for visibility
                2,
                cv2.LINE_AA
            )
            
            video_writer_int.write(cv_img)
            frame_id += 1

        print(f"Extracted {frame_id} frames from Internal Camera.")

        # Compute Tag stability: Ideally all tags should be detected at all frames. This would give stability = 1.

        total_frames = frame_id 
        with open('frames_output/Tag_stability_int.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['tag_id', 'stability_value'])
            total_stability = 0
            for tag_id, poses in tag_paths.items():
                stability = len(poses) / total_frames if total_frames > 0 else 0
                total_stability += stability
                writer.writerow([tag_id, stability])
            
            # Calculate and write the average stability
            avg_stability = total_stability / len(tag_paths) if tag_paths else 0
            writer.writerow(['Average', avg_stability])
        
        print("Stability metrics saved to tag_stability_int.csv")

    if video_writer_ext is not None:
        video_writer_ext.release()

    if video_writer_int is not None:
        video_writer_int.release()


# Building the video wiith both the frames side bu side can be done in a separate script if needed.


#######################################################################################
                    # Create Side-by-Side Combined Video
#######################################################################################


# Paths of your two individual videos
ext_video_path = os.path.join(output_dir, video_filename_Ext)
int_video_path = os.path.join(output_dir, video_filename_Int)

combined_video_path = os.path.join(output_dir, "combined_ext_int.avi")

# Open readers
ext_cap = cv2.VideoCapture(ext_video_path)
int_cap = cv2.VideoCapture(int_video_path)

# Check FPS / resolution of both videos [fps ext is greater than fos int in real life, but the video caoture changes it to 30]
# fps_ext = ext_cap.get(cv2.CAP_PROP_FPS)

# Need to manually sync the videos:
# Total time duration of each is same, so fps of each is duration/number of frames.
# Calculate FPS for both videos based on duration and frame count
# int_frame_count = int_cap.get(cv2.)

ext_frame_count = int(ext_cap.get(cv2.CAP_PROP_FRAME_COUNT))
int_frame_count = int(int_cap.get(cv2.CAP_PROP_FRAME_COUNT))




# Calculate video duration based on the time length of the internal video
fps_int = int_cap.get(cv2.CAP_PROP_FPS) # would be 30 only.
fps_ext = ext_cap.get(cv2.CAP_PROP_FPS)


# Output FPS (use min to avoid interpolation)
out_fps = min(fps_ext, fps_int)
if out_fps <= 0:
    out_fps = 30  # fallback

# print("External FPS:", out_fps)

# Initialize time trackers for synchronization
ext_w = int(ext_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ext_h = int(ext_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
int_w = int(int_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
int_h = int(int_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("External Video - Width:", ext_w, "Height:", ext_h)
# print("Internal Video - Width:", int_w, "Height:", int_h)


# Ensure height match — resize internal frame
target_h = max(ext_h, int_h)

# Combined video dimensions
combined_w = ext_w + int_w
combined_h = target_h

# print("Combined Video - Width:", combined_w, "Height:", combined_h)

# Using the larger number of frames to define output length
total_output_frames = max(ext_frame_count, int_frame_count)

combined_writer = cv2.VideoWriter(
    combined_video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    out_fps,  # Using combined video at slowest video speed.
    (combined_w, combined_h)
)

# print("Writer opened:", combined_writer.isOpened())



frame_idx = 0

for i in range(total_output_frames):

    # Compute aligned indices
    ext_idx = round(i * (ext_frame_count - 1) / (total_output_frames - 1))
    int_idx = round(i * (int_frame_count - 1) / (total_output_frames - 1))
    # print(f"Processing combined frame {i}: ext_idx={ext_idx}, int_idx={int_idx}")

    # Read external frame
    ext_cap.set(cv2.CAP_PROP_POS_FRAMES, ext_idx)          # WHYYYYYYYYYYYYYYYY
    # ext_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   
    ret_ext, ext_frame = ext_cap.read()
    if not ret_ext:
        print("Failed to read external frame, using black frame.")
        ext_frame = np.zeros((ext_h, ext_w, 3), dtype=np.uint8)

    # Read internal frame
    int_cap.set(cv2.CAP_PROP_POS_FRAMES, int_idx)
    # int_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_int, int_frame = int_cap.read()
    if not ret_int:
        print("Failed to read internal frame, using black frame.")
        int_frame = np.zeros((int_h, int_w, 3), dtype=np.uint8)

    # actual_ext = int(ext_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # actual_int = int(int_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # ext_frame_count = actual_ext
    # int_frame_count = actual_int
    

    # if i == 0:
    #     print("Trying to read EXT idx:", ext_idx)
    #     print("Trying to read INT idx:", int_idx)
    #     print("ret_ext:", ret_ext, "ret_int:", ret_int)

    # Resize internal frame if needed
    if int_frame.shape[0] != target_h:
        # print("Resizing INT frame from", int_frame.shape)
        scale = target_h / int_frame.shape[0]
        int_frame = cv2.resize(int_frame, int(int_frame.shape[1], target_h))
        # print("Resized INT frame to", int_frame.shape)

    if ext_frame.shape[0] != target_h:
        # print("Resizing EXT frame from", ext_frame.shape)
        scale = target_h / ext_frame.shape[0]
        ext_frame = cv2.resize(ext_frame, (ext_frame.shape[1] , target_h))
        # print("Resized EXT frame to", ext_frame.shape)

    
    # print("Actual EXT frames:", actual_ext)
    # print("Actual INT frames:", actual_int)

    # if i < 5 or i == total_output_frames - 1:
    #     print(
    #         f"[{i}] ext_idx={ext_idx}, ext_shape={ext_frame.shape}, "
    #         f"int_idx={int_idx}, int_shape={int_frame.shape}"
    #     )

    # print("Writer expects:", combined_w, combined_h)

    # Combine side-by-side
    combined = np.hstack((ext_frame, int_frame))
    combined_writer.write(combined)

print("total_output_frames =", total_output_frames)

print(f"Combined video created: {combined_video_path}")

# Cleanup
ext_cap.release()
int_cap.release()
combined_writer.release()


#######################################################################################
#                                  Completion Message
#######################################################################################

# print(f"Extracted {frame_id} frames and saved video to {os.path.join(output_dir, video_filename_Ext)}")
# print(f"Extracted {frame_id} frames and saved video to {os.path.join(output_dir, video_filename_Int)}")

print(f"Timestamps → {csv_filename_Ext}")
print(f"AprilTag poses ext → {tag_csv_filename_Ext}")
print(f"Timestamps → {csv_filename_Int}")
print(f"AprilTag poses int → {tag_csv_filename_Int}")


#######################################################################################
#                                  END OF FILE
#######################################################################################