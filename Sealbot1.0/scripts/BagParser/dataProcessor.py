# This file is intended to be used after the raw data is extracted from the bags by bagPaeser. 
# And we process the pose data of both internal and external camera sensors here to be ready for model training.

import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.spatial.transform import Rotation as R
import bisect
import pywt
import matplotlib.pyplot as plt
from collections import defaultdict  

#######################################################################################
#                         Configuration Parameters
#######################################################################################

# External Cube Dimensions
cube_size = 0.166  # meters     
static_tag_id = 19  # Mid of the wisker arrays in seal head. # 4: before ||  #19: new last one
tag_rel_poses = {}  

#######################################################################################
#               Int Tag Thresholding to prevent false positives
#######################################################################################

# Input is csv file and output is filtered tag_paths: but what about the frames where no tags are visible?: checksum sort logic in the perception model pipeline. 
# TODO: Edge Cases

# Tag pose reference with quaternion fixed: Calculated from the static_ref script.
internal_tag_id_rel_pose_map = {
    0: {"position": [0.01431607, 0.04001835, -0.07066817], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    1: {"position": [0.00856013, 0.03377553, -0.04614408], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    2: {"position": [0.01684475, 0.03069469, -0.01257934], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    3: {"position": [-0.00075084, -0.05271414, -0.01421607], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    5: {"position": [-0.01197729, -0.04446193, -0.0077042], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    6: {"position": [-0.0211849, -0.03541196, -0.00963364], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    7: {"position": [-0.01306488, -0.03531467, 0.01683272], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    8: {"position": [-0.02081204, -0.05283257, -0.03941675], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    9: {"position": [-0.01011581, -0.05187832, -0.03616606], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    10: {"position": [-0.00087129, -0.04338111, 0.00374754], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    11: {"position": [0.00564222, 0.02567888, -0.03249634], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    12: {"position": [0.02347466, 0.04053535, -0.07390323], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    13: {"position": [-0.02242572, -0.04503598, -0.01635032], "quaternion": [0.09239, 0.3827, 0.0, 0.0]},
    14: {"position": [0.01984042, 0.03416312, -0.05061772], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    16: {"position": [-0.00189823, 0.03547269, -0.03190359], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    17: {"position": [-0.00726191, 0.02756754, -0.01221053], "quaternion": [0.7071, 0.0, 0.7071, 0.0]},
    18: {"position": [0.00206168, 0.03940511, -0.06486038], "quaternion": [0.7071, 0.0, 0.7071, 0.0]}
}

margin_error_position = 0.06 # 6 cm
static_pos, static_quaternion = 0,0

def unfilter_rel_paths():
    # Read CSV and group by tag_id
    tag_poses = {}
    # with open('frames_output/tag_paths_int.csv', 'r') as f: 
    with open('frames_output/tag_paths_int.csv', 'r') as f: 
    
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            frame_id = int(row['frame_id'])
            tx = float(row['tx'])
            ty = float(row['ty'])
            tz = float(row['tz'])
            qx = float(row['qx'])
            qy = float(row['qy'])
            qz = float(row['qz'])
            qw = float(row['qw'])

            if frame_id not in tag_poses:
                tag_poses[frame_id] = {}
            tag_poses[frame_id][tag_id] = (np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))

    rel_tag_poses = {}

    # For fixed frame_id, this will have all "19" poses and will iterate next to the next frame_id.
    for frame_id, poses in tag_poses.items():
        if static_tag_id not in poses: # this continues with the previous static tag pose where the static tag is not detected.
            continue
            # static_pos, static_quaternion = static_pos, static_quaternion
        else:
            static_pos, static_quaternion = poses[static_tag_id]
            static_rot = R.from_quat(static_quaternion)

        for tag_id, (pos, quat) in poses.items():
            if tag_id == static_tag_id:
                continue

            # Check if tag_id is in the reference map
            if tag_id not in internal_tag_id_rel_pose_map:
                continue

            rel_pos = static_rot.inv().apply(pos - static_pos)
            rot = R.from_quat(quat)
            rel_rot = static_rot.inv() * rot

            # Use rel_pos and rel_rot for further processing.
            if frame_id not in rel_tag_poses:
                rel_tag_poses[frame_id] = {}
            rel_tag_poses[frame_id][tag_id] = (rel_pos, rel_rot)
        
    return rel_tag_poses

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

'''
Calculate the robust Z-score of a 1D array using Median and Median Absolute Deviation (MAD).
'''
def z_score(values):
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    mad = max(mad, 1e-12)  # avoid /0
    return 0.6745 * np.abs(values - med) / mad  # normalized abs deviation

# TODO: Check this function.
def filter_1d_signal(sig, use_angle_wrap = True, z_thresh=5.5):
    sig = np.asarray(sig).copy()
    N = len(sig)

    # Compute first differences (Δx)
    if use_angle_wrap:
        d = wrap_angle(np.diff(sig))
    else:
        d = np.diff(sig)

    # Robust Z-score/
    z = z_score(d)

    # Spike indices are where jump is too big
    spike_idx = np.where(z > z_thresh)[0] + 1   # +1 aligns to original signal index

    # Replace spike points using linear interpolation
    for i in spike_idx:
        if i == 0 or i == N-1:
            continue
        sig[i] = 0.5*(sig[i-1] + sig[i+1])

    return sig


# scipy - find peeks
# filter 1: based on physical info : then do a moving average filter. 
# moving avg filter will ensure that if peeks are supposed to be high in a range, it wont be removed, but this needs the physical filter first.
# I have a pretty low frequency data (8-11 fps == Hz), so we cant do too much smoothing or low pass filtering.


def filter_spikes_1d_signal(signal, percentage_threshold=0.50):
    # Calculate the mean and variance of the signal
    # Remove the outlier values with percentage thresholding
    # This keeps the middle percentage_threshold of values.

    signal = np.asarray(signal)
    N = len(signal)

    # Compute symmetric low/high percentile limits
    low_cut = np.percentile(signal, 100 * (1 - percentage_threshold) / 2)
    high_cut = np.percentile(signal, 100 * (1 + percentage_threshold) / 2)

    filtered_signal = signal.copy()

    # Loop and replace outliers with the average of neighbors
    for i in range(1, N - 1):
        if filtered_signal[i] < low_cut or filtered_signal[i] > high_cut:
            filtered_signal[i] = 0.5 * (filtered_signal[i - 1] + filtered_signal[i + 1])

    return filtered_signal

def filter_false_positive():
    # Read CSV and group by tag_id
    tag_poses = {}
    with open('frames_output/tag_paths_int.csv', 'r') as f: 
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            frame_id = int(row['frame_id'])
            tx = float(row['tx'])
            ty = float(row['ty'])
            tz = float(row['tz'])
            qx = float(row['qx'])
            qy = float(row['qy'])
            qz = float(row['qz'])
            qw = float(row['qw'])

            if frame_id not in tag_poses:
                tag_poses[frame_id] = {}
            tag_poses[frame_id][tag_id] = (np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))

    rel_tag_poses = {}

    # For fixed frame_id, this will have all "19" poses and will iterate next to the next frame_id.
    for frame_id, poses in tag_poses.items():
        if static_tag_id not in poses: # this discards all frames where the static tag is not detected.
            continue

        static_pos, static_quaternion = poses[static_tag_id]
        static_rot = R.from_quat(static_quaternion)

        for tag_id, (pos, quat) in poses.items():
            if tag_id == static_tag_id:
                continue

            # Check if tag_id is in the reference map
            if tag_id not in internal_tag_id_rel_pose_map:
                continue

            rel_pos = static_rot.inv().apply(pos - static_pos)
            rot = R.from_quat(quat)
            rel_rot = static_rot.inv() * rot

            # Check if the relative position is within margin error from the reference position.
            ref_pos = np.array(internal_tag_id_rel_pose_map[tag_id]["position"])
            if np.linalg.norm(rel_pos - ref_pos) > margin_error_position:
                continue

            # # Check if the relative rotation is within margin error from the reference quaternion.
            # ref_quat = np.array(internal_tag_id_rel_pose_map[tag_id]["quaternion"])
            # ref_rot = R.from_quat(ref_quat)
            # if rel_rot.inv() * ref_rot.magnitude() > margin_error_position:
            #     continue

            # Use rel_pos and rel_rot for further processing.
            if frame_id not in rel_tag_poses:
                rel_tag_poses[frame_id] = {}
            rel_tag_poses[frame_id][tag_id] = (rel_pos, rel_rot)
        
    return rel_tag_poses

#######################################################################################
#          Plot the Relative Tag Pose for all Internal Tags wrt Timestamp
#######################################################################################

# # Relative poses global variable
# tx_rel = []
# ty_rel = []
# tz_rel = []

'''
Calculate the filtered signals for multiple specific tags from the relative tag poses.
'''
def get_filtered_signals(rel_tag_poses, tagA=None, tagB=17, tagC=14, tagD=None):

    frame_idxs = sorted(rel_tag_poses.keys())

    tx_A, ty_A, tz_A, roll_A, pitch_A, yaw_A = [], [], [], [], [], []
    tx_B, ty_B, tz_B, roll_B, pitch_B, yaw_B = [], [], [], [], [], []
    tx_C, ty_C, tz_C, roll_C, pitch_C, yaw_C = [], [], [], [], [], []
    tx_D, ty_D, tz_D, roll_D, pitch_D, yaw_D = [], [], [], [], [], []

    default_value = 0
    
    # Calculate the starting positions for displacement calculation
    # Right now I am doing relative to the first detected pose.

    # TODO: Assuming the first frame has all tags visible: Necessary for this to work.
    start_pos = {}
    poses = rel_tag_poses[frame_idxs[0]]
    if tagA in poses:
        # pos_A, rot_A = poses[tagA]
        start_pos[tagA] = poses[tagA]
    if tagB in poses:
        # pos_B, _ = poses[tagB]
        start_pos[tagB] = poses[tagB]
    if tagC in poses:
        # pos_C, _ = poses[tagC]
        start_pos[tagC] = poses[tagC]
    if tagD in poses:
        # pos_D, _ = poses[tagD]
        start_pos[tagD] = poses[tagD]

    for frame_idx in frame_idxs:
        poses = rel_tag_poses[frame_idx]
        if tagA in poses:
            pos_A, quat_A = calculate_rel_pose(poses[tagA], start_pos[tagA])
            tx_A.append(pos_A[0])
            ty_A.append(pos_A[1])
            tz_A.append(pos_A[2])
            r = R.from_quat(quat_A)  
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_A.append(roll)
            pitch_A.append(pitch)
            yaw_A.append(yaw)
        else:
            tx_A.append(default_value)
            ty_A.append(default_value)
            tz_A.append(default_value)
            roll_A.append(default_value)
            pitch_A.append(default_value)
            yaw_A.append(default_value)
        if tagB in poses:
            pos_B, quat_B = calculate_rel_pose(poses[tagB], start_pos[tagB])
            tx_B.append(pos_B[0])
            ty_B.append(pos_B[1])
            tz_B.append(pos_B[2])
            r = R.from_quat(quat_B)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_B.append(roll)
            pitch_B.append(pitch)
            yaw_B.append(yaw)

        else:
            tx_B.append(default_value)
            ty_B.append(default_value)
            tz_B.append(default_value)
            roll_B.append(default_value)
            pitch_B.append(default_value)
            yaw_B.append(default_value)
        if tagC in poses:
            pos_C, quat_C = calculate_rel_pose(poses[tagC], start_pos[tagC])
            tx_C.append(pos_C[0])
            ty_C.append(pos_C[1])
            tz_C.append(pos_C[2])
            r = R.from_quat(quat_C)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_C.append(roll)
            pitch_C.append(pitch)
            yaw_C.append(yaw)
        else:
            tx_C.append(default_value)
            ty_C.append(default_value)
            tz_C.append(default_value)
            roll_C.append(default_value)
            pitch_C.append(default_value)
            yaw_C.append(default_value)
        if tagD in poses:
            pos_D, quat_D = calculate_rel_pose(poses[tagD], start_pos[tagD])
            tx_D.append(pos_D[0])
            ty_D.append(pos_D[1])
            tz_D.append(pos_D[2])
            r = R.from_quat(quat_D)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_D.append(roll)
            pitch_D.append(pitch)
            yaw_D.append(yaw)
        else:   
            tx_D.append(default_value)
            ty_D.append(default_value)
            tz_D.append(default_value)
            roll_D.append(default_value)
            pitch_D.append(default_value)
            yaw_D.append(default_value)

    # Filtering by removing physical impossible points. 
    tx_A = physical_signal_filter_translation(tx_A)
    ty_A = physical_signal_filter_translation(ty_A)
    tz_A = physical_signal_filter_translation(tz_A)
    tx_B = physical_signal_filter_translation(tx_B)
    ty_B = physical_signal_filter_translation(ty_B)
    tz_B = physical_signal_filter_translation(tz_B)
    tx_C = physical_signal_filter_translation(tx_C)
    ty_C = physical_signal_filter_translation(ty_C)
    tz_C = physical_signal_filter_translation(tz_C)
    tx_D = physical_signal_filter_translation(tx_D)
    ty_D = physical_signal_filter_translation(ty_D)
    tz_D = physical_signal_filter_translation(tz_D)

    roll_A = physical_signal_filter_rotation(roll_A)
    pitch_A = physical_signal_filter_rotation(pitch_A)
    yaw_A = physical_signal_filter_rotation(yaw_A)
    roll_B = physical_signal_filter_rotation(roll_B)
    pitch_B = physical_signal_filter_rotation(pitch_B)
    yaw_B = physical_signal_filter_rotation(yaw_B)
    roll_C = physical_signal_filter_rotation(roll_C)
    pitch_C = physical_signal_filter_rotation(pitch_C)
    yaw_C = physical_signal_filter_rotation(yaw_C)
    roll_D = physical_signal_filter_rotation(roll_D)
    pitch_D = physical_signal_filter_rotation(pitch_D)
    yaw_D = physical_signal_filter_rotation(yaw_D)

    filtered_signals = defaultdict(dict)
    if tagA is not None:
        filtered_signals[tagA]['x'] = tx_A
        filtered_signals[tagA]['y'] = ty_A
        filtered_signals[tagA]['z'] = tz_A
        filtered_signals[tagA]['roll'] = roll_A
        filtered_signals[tagA]['pitch'] = pitch_A
        filtered_signals[tagA]['yaw'] = yaw_A
    if tagB is not None:
        filtered_signals[tagB]['x'] = tx_B
        filtered_signals[tagB]['y'] = ty_B
        filtered_signals[tagB]['z'] = tz_B
        filtered_signals[tagB]['roll'] = roll_B
        filtered_signals[tagB]['pitch'] = pitch_B
        filtered_signals[tagB]['yaw'] = yaw_B
    if tagC is not None:
        filtered_signals[tagC]['x'] = tx_C
        filtered_signals[tagC]['y'] = ty_C
        filtered_signals[tagC]['z'] = tz_C
        filtered_signals[tagC]['roll'] = roll_C
        filtered_signals[tagC]['pitch'] = pitch_C
        filtered_signals[tagC]['yaw'] = yaw_C
    if tagD is not None:
        filtered_signals[tagD]['x'] = tx_D
        filtered_signals[tagD]['y'] = ty_D
        filtered_signals[tagD]['z'] = tz_D
        filtered_signals[tagD]['roll'] = roll_D
        filtered_signals[tagD]['pitch'] = pitch_D
        filtered_signals[tagD]['yaw'] = yaw_D

    return filtered_signals
    
'''
Plot the 3D relative pose of all internal tags with respect to a static tag (ID = static_tag_id) over time.
'''
def plot_3d_relative_pose(ax_relative):
    ax_relative.clear()  # Clear the previous plot
    # Read CSV and group by tag_id
    tag_paths = {}
    with open('frames_output/tag_paths_int.csv', 'r') as f: 
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(tag_id, []).append(pose)

    # Extract pose with ID = 0 as reference
    if static_tag_id not in tag_paths:
        print("No tag with ID static_tag_id found for reference.")
        return
    ref_poses = sorted(tag_paths[0], key=lambda x: x[0])  # sort by frame_idx

    # Plot each tag's path relative to tag 0: as in x-x1, y-y1, z-z1

    for tag_id, poses in tag_paths.items():
        if tag_id == static_tag_id:
            continue
        poses = sorted(poses, key=lambda x: x[0])
        tx_rel = []
        ty_rel = []
        tz_rel = []

        for p in poses:
            frame_idx = p[0]
            ref_pose = next((rp for rp in ref_poses if rp[0] == frame_idx), None)
            if ref_pose is None:
                continue

            # Extract translation poses
            t = np.array([p[1], p[2], p[3]])
            t_ref = np.array([ref_pose[1], ref_pose[2], ref_pose[3]])
            # Extract rotation poses
            R = R.from_quat(p[4:8]).as_matrix()
            R_ref = R.from_quat(ref_pose[4:8]).as_matrix()

            # Compute relative transformation
            R_rel =  R_ref.T @ R
            t_rel = R_ref.T @ (t - t_ref)

            tx_rel.append(t_rel[0])
            ty_rel.append(t_rel[1])
            tz_rel.append(t_rel[2])      
        
        ax_relative.plot(tx_rel, ty_rel, tz_rel, label=f'Tag {tag_id}')
    
    ax_relative.set_xlabel('X rel to Static Tag')
    ax_relative.set_ylabel('Y rel to Static Tag')
    ax_relative.set_zlabel('Z rel to Static Tag')
    ax_relative.legend()
    ax_relative.set_title('Relative 3D Poses to Static Tag')
    plt.show(block=False)
    plt.pause(0.01)

'''
Plot the 3D relative pose of a specific tag relative to Static Tag over time. 
'''
def plot_relative_pose_indv(tag_id, ax_indv):
    ax_indv.clear()  # Clear the previous plot
    tag_paths = {}
    with open('frames_output/tag_paths_int.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if static_tag_id not in tag_paths or tag_id not in tag_paths:
        print(f"Static Tag or Tag {tag_id} not found.")
        return
    ref_poses = sorted(tag_paths[0], key=lambda x: x[0])
    poses = sorted(tag_paths[tag_id], key=lambda x: x[0])
    tx_rel = []
    ty_rel = []
    tz_rel = []

    for p in poses:
        frame_idx = p[0]
        ref_pose = next((rp for rp in ref_poses if rp[0] == frame_idx), None)
        if ref_pose is None:
            continue

        # Extract translation poses
        t = np.array([p[1], p[2], p[3]])
        t_ref = np.array([ref_pose[1], ref_pose[2], ref_pose[3]])
        # Extract rotation poses
        R_stat = R.from_quat(p[4:8]).as_matrix()
        R_ref = R.from_quat(ref_pose[4:8]).as_matrix()

        # Compute relative transformation
        R_rel =  R_ref.T @ R_stat
        t_rel = R_ref.T @ (t - t_ref)

        tx_rel.append(t_rel[0])
        ty_rel.append(t_rel[1])
        tz_rel.append(t_rel[2])
    
    
    ax_indv.plot(tx_rel, ty_rel, tz_rel, label=f'Relative Path of Tag {tag_id}')
    ax_indv.set_xlabel('X rel to Static Tag')
    ax_indv.set_ylabel('Y rel to Static Tag')
    ax_indv.set_zlabel('Z rel to Static Tag')
    ax_indv.legend()
    ax_indv.set_title(f'Relative Path of Tag {tag_id} to Static Tag')
    plt.show()

'''
Plot the individual x, y, z signals of a specific tag separately with respect to a static tag (ID = static_tag_id) over time.
'''
def plot_idv(tag_id, ax_x = None, ax_y= None, ax_z= None):
    # ax_indv_axis.clear()  # Clear the previous plot
    tag_paths = {}
    with open('frames_output/tag_paths_int.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if tag_id not in tag_paths:
        print(f"Tag {tag_id} not found.")
        return

    poses = sorted(tag_paths[tag_id], key=lambda x: x[0])
    frame_idxs = [p[0] for p in poses]
    tx = [p[1] for p in poses]
    ty = [p[2] for p in poses]
    tz = [p[3] for p in poses]

    # Relative to static tag.
    if static_tag_id in tag_paths:
        static_poses = sorted(tag_paths[static_tag_id], key=lambda x: x[0])
        static_dict = {p[0]: p for p in static_poses}
        tx_rel = []
        ty_rel = []
        tz_rel = []
        for p in poses:
            frame_idx = p[0]
            if frame_idx in static_dict:
                static_p = static_dict[frame_idx]
                tx_rel.append(p[1] - static_p[1])
                ty_rel.append(p[2] - static_p[2])
                tz_rel.append(p[3] - static_p[3])
            else:
                tx_rel.append(p[1])
                ty_rel.append(p[2])
                tz_rel.append(p[3])
        tx, ty, tz = tx_rel, ty_rel, tz_rel

    # Plot X signal
    ax_x.plot(frame_idxs, tx, label='X', color='r')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y signal
    ax_y.plot(frame_idxs, ty, label='Y', color='g')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z signal
    ax_z.plot(frame_idxs, tz, label='Z', color='b')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

'''
Plot the individual x, y, z signals of a specific tag relative to another specific tag over time.
'''
def plot_idv_comparative(tag_id_1, tag_id_2, ax_indv_axis, ax_x = None, ax_y= None, ax_z= None):
    # Plot the individual x, y, z signals of a specific tag separately over time.
    ax_indv_axis.clear()  # Clear the previous plot
    tag_paths = {}
    with open('tag_paths.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if tag_id_1 not in tag_paths or tag_id_2 not in tag_paths:
        print(f"Tag {tag_id_1} or Static Tag {tag_id_2} not found.")
        return
    tag_id1_poses = sorted(tag_paths[tag_id_1], key=lambda x: x[0])
    tag_id2_poses = sorted(tag_paths[tag_id_2], key=lambda x: x[0])

    frame_idxs = []
    tx_rel = []
    ty_rel = []
    tz_rel = []

    for p in tag_id1_poses:
        frame_idx = p[0]
        tag_id1_pose = next((sp for sp in tag_id1_poses if sp[0] == frame_idx), None)
        tag_id2_pose = next((sp for sp in tag_id2_poses if sp[0] == frame_idx), None)

        if tag_id2_pose is None:
            continue
        tag_1 = np.array([tag_id1_pose[1], tag_id1_pose[2], tag_id1_pose[3]])
        tag_2 = np.array([tag_id2_pose[1], tag_id2_pose[2], tag_id2_pose[3]])
        frame_idxs.append(frame_idx)
        tx_rel.append(tag_1[0] - tag_2[0])
        ty_rel.append(tag_1[1] - tag_2[1])
        tz_rel.append(tag_1[2] - tag_2[2])

    # Plot X signal
    ax_x.plot(frame_idxs, tx_rel, label='X rel to Static Tag', color='r')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y signal
    ax_y.plot(frame_idxs, ty_rel, label='Y rel to Static Tag', color='g')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z signal
    ax_z.plot(frame_idxs, tz_rel, label='Z rel to Static Tag', color='b')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # plt.pause(0.01)

'''
Plot the individual x, y, z signals of multiple specific filtered tags over time.
'''
def plot_filtered_tags(rel_tag_poses, ax_x, ax_y, ax_z, tagA=2, tagB=4, tagC=5, tagD=5):
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()

    frame_idxs = sorted(rel_tag_poses.keys())
    tx_A, ty_A, tz_A = [], [], []
    tx_B, ty_B, tz_B = [], [], []
    tx_C, ty_C, tz_C = [], [], []
    tx_D, ty_D, tz_D = [], [], []

    default_value = 0

    for frame_idx in frame_idxs:
        poses = rel_tag_poses[frame_idx]
        if tagA in poses:
            pos_A, _ = poses[tagA]
            tx_A.append(pos_A[0])
            ty_A.append(pos_A[1])
            tz_A.append(pos_A[2])
        else:
            tx_A.append(default_value)
            ty_A.append(default_value)
            tz_A.append(default_value)
        if tagB in poses:
            pos_B, _ = poses[tagB]
            tx_B.append(pos_B[0])
            ty_B.append(pos_B[1])
            tz_B.append(pos_B[2])
        else:
            tx_B.append(default_value)
            ty_B.append(default_value)
            tz_B.append(default_value)
        if tagC in poses:
            pos_C, _ = poses[tagC]
            tx_C.append(pos_C[0])
            ty_C.append(pos_C[1])
            tz_C.append(pos_C[2])
        else:
            tx_C.append(default_value)
            ty_C.append(default_value)
            tz_C.append(default_value)
        if tagD in poses:
            pos_D, _ = poses[tagD]
            tx_D.append(pos_D[0])
            ty_D.append(pos_D[1])
            tz_D.append(pos_D[2])
        else:   
            tx_D.append(default_value)
            ty_D.append(default_value)
            tz_D.append(default_value)

    # Plot X: TODO: need to fix the lengths if some tags are missing in some frames.
    # Also separate each tag frame idxs plots.
    ax_x.plot(tx_A, label=f'Tag {tagA}', color='r')
    ax_x.plot(tx_B, label=f'Tag {tagB}', color='g')
    ax_x.plot(tx_C, label=f'Tag {tagC}', color='b')
    ax_x.plot(tx_D, label=f'Tag {tagD}', color='m')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y
    ax_y.plot(ty_A, label=f'Tag {tagA}', color='r')
    ax_y.plot(ty_B, label=f'Tag {tagB}', color='g')
    ax_y.plot(ty_C, label=f'Tag {tagC}', color='b')
    ax_y.plot(ty_D, label=f'Tag {tagD}', color='m')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z
    ax_z.plot( tz_A, label=f'Tag {tagA}', color='r')
    ax_z.plot(tz_B, label=f'Tag {tagB}', color='g')
    ax_z.plot(tz_C, label=f'Tag {tagC}', color='b')
    ax_z.plot(tz_D, label=f'Tag {tagD}', color='m')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

"""
    Calculate the relative pose from the start_pose to the current_pose.
    Both poses are given as (position, rotation) tuples.
"""
def calculate_rel_pose(current_pose, start_pose):
    
    curr_pos, curr_rot = current_pose
    start_pos, start_rot = start_pose

    # Calculate relative rotation
    rel_rot = start_rot.inv() * curr_rot
    rel_quat = rel_rot.as_quat()  # (x, y, z, w)

    # Calculate relative position
    rel_pos = start_rot.inv().apply(curr_pos - start_pos)

    return (rel_pos, rel_quat)

'''
Alll x,y,z signals should be within physical limits, if not setting them to interpolated values.
'''
def physical_signal_filter_translation(signal):
    sig = np.asarray(signal)
    N = len(sig)
    # Define physical limits
    physical_limit = 0.0153  # 15.3 mm
    # Loop and replace outliers with the average of neighbors
    for i in range(1, N - 1):
        if abs(sig[i]) > physical_limit:
            sig[i] = 0.5 * (sig[i - 1] + sig[i - 2])
    return sig

def physical_signal_filter_rotation(signal):
    sig = np.asarray(signal)
    N = len(sig)
    # Define physical limits
    physical_limit = np.radians(7.3) # np.radians(7.3) == 0.127 
    # Limit testing: 10.83 degrees in radians | 5.8 | 6.5 
    # Loop and replace outliers with the average of neighbors
    for i in range(1, N - 1):
        if abs(sig[i]) > physical_limit:
            sig[i] = 0.5 * (sig[i - 1] + sig[i - 2])
    return sig

'''
Plot the individual x, y, z, roll, pitch, yaw signals of multiple specific UN-filtered tags over time: Displcement from initial position.
'''
def plot_unfiltered_tags_displacement(rel_tag_poses, ax_x, ax_y, ax_z, ax_roll, ax_pitch, ax_yaw, tagA=None, tagB=None, tagC=None, tagD=None):
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()

    frame_idxs = sorted(rel_tag_poses.keys())

    tx_A, ty_A, tz_A, roll_A, pitch_A, yaw_A = [], [], [], [], [], []
    tx_B, ty_B, tz_B, roll_B, pitch_B, yaw_B = [], [], [], [], [], []
    tx_C, ty_C, tz_C, roll_C, pitch_C, yaw_C = [], [], [], [], [], []
    tx_D, ty_D, tz_D, roll_D, pitch_D, yaw_D = [], [], [], [], [], []

    default_value = 0

    # Calculate the starting positions for displacement calculation.
    # Tag pose relative to the first detected pose.
    # TODO: Assuming the first frame has all tags visible: Necessary for this to work.
    start_pos = {}
    poses = rel_tag_poses[frame_idxs[0]]
    if tagA in poses:
        # pos_A, rot_A = poses[tagA]
        start_pos[tagA] = poses[tagA]
    if tagB in poses:
        # pos_B, _ = poses[tagB]
        start_pos[tagB] = poses[tagB]
    if tagC in poses:
        # pos_C, _ = poses[tagC]
        start_pos[tagC] = poses[tagC]
    if tagD in poses:
        # pos_D, _ = poses[tagD]
        start_pos[tagD] = poses[tagD]

    for frame_idx in frame_idxs:
        poses = rel_tag_poses[frame_idx]
        if tagA in poses:
            pos_A, quat_A = calculate_rel_pose(poses[tagA], start_pos[tagA])
            tx_A.append(pos_A[0])
            ty_A.append(pos_A[1])
            tz_A.append(pos_A[2])
            r = R.from_quat(quat_A)  
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_A.append(roll)
            pitch_A.append(pitch)
            yaw_A.append(yaw)
        else:
            tx_A.append(default_value)
            ty_A.append(default_value)
            tz_A.append(default_value)
            roll_A.append(default_value)
            pitch_A.append(default_value)
            yaw_A.append(default_value)
        if tagB in poses:
            pos_B, quat_B = calculate_rel_pose(poses[tagB], start_pos[tagB])
            tx_B.append(pos_B[0])
            ty_B.append(pos_B[1])
            tz_B.append(pos_B[2])
            r = R.from_quat(quat_B)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_B.append(roll)
            pitch_B.append(pitch)
            yaw_B.append(yaw)

        else:
            tx_B.append(default_value)
            ty_B.append(default_value)
            tz_B.append(default_value)
            roll_B.append(default_value)
            pitch_B.append(default_value)
            yaw_B.append(default_value)
        if tagC in poses:
            pos_C, quat_C = calculate_rel_pose(poses[tagC], start_pos[tagC])
            tx_C.append(pos_C[0])
            ty_C.append(pos_C[1])
            tz_C.append(pos_C[2])
            r = R.from_quat(quat_C)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_C.append(roll)
            pitch_C.append(pitch)
            yaw_C.append(yaw)
        else:
            tx_C.append(default_value)
            ty_C.append(default_value)
            tz_C.append(default_value)
            roll_C.append(default_value)
            pitch_C.append(default_value)
            yaw_C.append(default_value)
        if tagD in poses:
            pos_D, quat_D = calculate_rel_pose(poses[tagD], start_pos[tagD])
            tx_D.append(pos_D[0])
            ty_D.append(pos_D[1])
            tz_D.append(pos_D[2])
            r = R.from_quat(quat_D)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_D.append(roll)
            pitch_D.append(pitch)
            yaw_D.append(yaw)
        else:   
            tx_D.append(default_value)
            ty_D.append(default_value)
            tz_D.append(default_value)
            roll_D.append(default_value)
            pitch_D.append(default_value)
            yaw_D.append(default_value)


    # tx_A = physical_signal_filter_translation(tx_A)
    # ty_A = physical_signal_filter_translation(ty_A)
    # tz_A = physical_signal_filter_translation(tz_A)
    # tx_B = physical_signal_filter_translation(tx_B)
    # ty_B = physical_signal_filter_translation(ty_B)
    # tz_B = physical_signal_filter_translation(tz_B)
    # tx_C = physical_signal_filter_translation(tx_C)
    # ty_C = physical_signal_filter_translation(ty_C)
    # tz_C = physical_signal_filter_translation(tz_C)
    # tx_D = physical_signal_filter_translation(tx_D)
    # ty_D = physical_signal_filter_translation(ty_D)
    # tz_D = physical_signal_filter_translation(tz_D)

    # roll_A = physical_signal_filter_rotation(roll_A)
    # pitch_A = physical_signal_filter_rotation(pitch_A)
    # yaw_A = physical_signal_filter_rotation(yaw_A)
    # roll_B = physical_signal_filter_rotation(roll_B)
    # pitch_B = physical_signal_filter_rotation(pitch_B)
    # yaw_B = physical_signal_filter_rotation(yaw_B)
    # roll_C = physical_signal_filter_rotation(roll_C)
    # pitch_C = physical_signal_filter_rotation(pitch_C)
    # yaw_C = physical_signal_filter_rotation(yaw_C)
    # roll_D = physical_signal_filter_rotation(roll_D)
    # pitch_D = physical_signal_filter_rotation(pitch_D)
    # yaw_D = physical_signal_filter_rotation(yaw_D)
    


    # Plot X: TODO: need to fix the lengths if some tags are missing in some frames.
    # Also separate each tag frame idxs plots.
    ax_x.plot(tx_A, label=f'Tag {tagA}', color='r')
    ax_x.plot(tx_B, label=f'Tag {tagB}', color='g')
    ax_x.plot(tx_C, label=f'Tag {tagC}', color='b')
    ax_x.plot(tx_D, label=f'Tag {tagD}', color='y')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X (m) Displacement from its intial pose')
    ax_x.legend()
    ax_x.grid()

    # Plot Y
    ax_y.plot(ty_A, label=f'Tag {tagA}', color='r')
    ax_y.plot(ty_B, label=f'Tag {tagB}', color='g')
    ax_y.plot(ty_C, label=f'Tag {tagC}', color='b')
    ax_y.plot(ty_D, label=f'Tag {tagD}', color='y')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y (m) Displacement from its intial pose')
    ax_y.legend()
    ax_y.grid()

    # Plot Z
    ax_z.plot(tz_A, label=f'Tag {tagA}', color='r')
    ax_z.plot(tz_B, label=f'Tag {tagB}', color='g')
    ax_z.plot(tz_C, label=f'Tag {tagC}', color='b')
    ax_z.plot(tz_D, label=f'Tag {tagD}', color='y')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z (m) Displacement from its intial pose')
    ax_z.legend()
    ax_z.grid()

    # Plot Roll
    ax_roll.plot(roll_A, label=f'Tag {tagA}', color='r')
    ax_roll.plot(roll_B, label=f'Tag {tagB}', color='g')
    ax_roll.plot(roll_C, label=f'Tag {tagC}', color='b')
    ax_roll.plot(roll_D, label=f'Tag {tagD}', color='y')
    ax_roll.set_xlabel('Frame Index')
    ax_roll.set_ylabel('Roll (rad) Displacement from its intial pose')
    ax_roll.legend()
    ax_roll.grid()

    # Plot Pitch
    ax_pitch.plot(pitch_A, label=f'Tag {tagA}', color='r')
    ax_pitch.plot(pitch_B, label=f'Tag {tagB}', color='g')
    ax_pitch.plot(pitch_C, label=f'Tag {tagC}', color='b')
    ax_pitch.plot(pitch_D, label=f'Tag {tagD}', color='y')
    ax_pitch.set_xlabel('Frame Index')
    ax_pitch.set_ylabel('Pitch (rad) Displacement from its intial pose')
    ax_pitch.legend()
    ax_pitch.grid()

    # Plot Yaw
    ax_yaw.plot(yaw_A, label=f'Tag {tagA}', color='r')
    ax_yaw.plot(yaw_B, label=f'Tag {tagB}', color='g')
    ax_yaw.plot(yaw_C, label=f'Tag {tagC}', color='b')
    ax_yaw.plot(yaw_D, label=f'Tag {tagD}', color='y')
    ax_yaw.set_xlabel('Frame Index')
    ax_yaw.set_ylabel('Yaw (rad) Displacement from its intial pose')
    ax_yaw.legend()
    ax_yaw.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
 
'''
Plot the individual x, y, z, roll, pitch, yaw signals of multiple specific spike filtered tags over time: Displcement from initial position.
'''
def plot_spike_filtered_tags_displacement(rel_tag_poses, ax_x, ax_y, ax_z, ax_roll, ax_pitch, ax_yaw, tagA=None, tagB=None, tagC=None, tagD=None):
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()

    frame_idxs = sorted(rel_tag_poses.keys())

    tx_A, ty_A, tz_A, roll_A, pitch_A, yaw_A = [], [], [], [], [], []
    tx_B, ty_B, tz_B, roll_B, pitch_B, yaw_B = [], [], [], [], [], []
    tx_C, ty_C, tz_C, roll_C, pitch_C, yaw_C = [], [], [], [], [], []
    tx_D, ty_D, tz_D, roll_D, pitch_D, yaw_D = [], [], [], [], [], []

    default_value = 0

    # Calculate the starting positions for displacement calculation
    # TODO: need to check if i want all pose relative to the first tag pose, or if I want pose relative to the previos frame pose. 
    # Right now I am doing relative to the first detected pose.

    # TODO: Assuming the first frame has all tags visible: Necessary for this to work.
    start_pos = {}
    poses = rel_tag_poses[frame_idxs[0]]
    if tagA in poses:
        # pos_A, rot_A = poses[tagA]
        start_pos[tagA] = poses[tagA]
    if tagB in poses:
        # pos_B, _ = poses[tagB]
        start_pos[tagB] = poses[tagB]
    if tagC in poses:
        # pos_C, _ = poses[tagC]
        start_pos[tagC] = poses[tagC]
    if tagD in poses:
        # pos_D, _ = poses[tagD]
        start_pos[tagD] = poses[tagD]

    for frame_idx in frame_idxs:
        poses = rel_tag_poses[frame_idx]
        if tagA in poses:
            pos_A, quat_A = calculate_rel_pose(poses[tagA], start_pos[tagA])
            tx_A.append(pos_A[0])
            ty_A.append(pos_A[1])
            tz_A.append(pos_A[2])
            r = R.from_quat(quat_A)  
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_A.append(roll)
            pitch_A.append(pitch)
            yaw_A.append(yaw)
        else:
            tx_A.append(default_value)
            ty_A.append(default_value)
            tz_A.append(default_value)
            roll_A.append(default_value)
            pitch_A.append(default_value)
            yaw_A.append(default_value)
        if tagB in poses:
            pos_B, quat_B = calculate_rel_pose(poses[tagB], start_pos[tagB])
            tx_B.append(pos_B[0])
            ty_B.append(pos_B[1])
            tz_B.append(pos_B[2])
            r = R.from_quat(quat_B)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_B.append(roll)
            pitch_B.append(pitch)
            yaw_B.append(yaw)

        else:
            tx_B.append(default_value)
            ty_B.append(default_value)
            tz_B.append(default_value)
            roll_B.append(default_value)
            pitch_B.append(default_value)
            yaw_B.append(default_value)
        if tagC in poses:
            pos_C, quat_C = calculate_rel_pose(poses[tagC], start_pos[tagC])
            tx_C.append(pos_C[0])
            ty_C.append(pos_C[1])
            tz_C.append(pos_C[2])
            r = R.from_quat(quat_C)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_C.append(roll)
            pitch_C.append(pitch)
            yaw_C.append(yaw)
        else:
            tx_C.append(default_value)
            ty_C.append(default_value)
            tz_C.append(default_value)
            roll_C.append(default_value)
            pitch_C.append(default_value)
            yaw_C.append(default_value)
        if tagD in poses:
            pos_D, quat_D = calculate_rel_pose(poses[tagD], start_pos[tagD])
            tx_D.append(pos_D[0])
            ty_D.append(pos_D[1])
            tz_D.append(pos_D[2])
            r = R.from_quat(quat_D)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_D.append(roll)
            pitch_D.append(pitch)
            yaw_D.append(yaw)
        else:   
            tx_D.append(default_value)
            ty_D.append(default_value)
            tz_D.append(default_value)
            roll_D.append(default_value)
            pitch_D.append(default_value)
            yaw_D.append(default_value)

    # Filter the signals to remove spikes
    # tx_A = filter_1d_signal(tx_A, use_angle_wrap=False)
    # ty_A = filter_1d_signal(ty_A, use_angle_wrap=False)
    # tz_A = filter_1d_signal(tz_A, use_angle_wrap=False)
    # roll_A = filter_1d_signal(roll_A, use_angle_wrap=True)
    # pitch_A = filter_1d_signal(pitch_A, use_angle_wrap=True)
    # yaw_A = filter_1d_signal(yaw_A, use_angle_wrap=True)

    tx_A = filter_spikes_1d_signal(tx_A)
    ty_A = filter_spikes_1d_signal(ty_A)
    tz_A = filter_spikes_1d_signal(tz_A)
    roll_A = filter_spikes_1d_signal(roll_A)
    pitch_A = filter_spikes_1d_signal(pitch_A)
    yaw_A = filter_spikes_1d_signal(yaw_A)
    tx_B = filter_spikes_1d_signal(tx_B)
    ty_B = filter_spikes_1d_signal(ty_B)
    tz_B = filter_spikes_1d_signal(tz_B)
    roll_B = filter_spikes_1d_signal(roll_B)
    pitch_B = filter_spikes_1d_signal(pitch_B)
    yaw_B = filter_spikes_1d_signal(yaw_B)
    tx_C = filter_spikes_1d_signal(tx_C)
    ty_C = filter_spikes_1d_signal(ty_C)
    tz_C = filter_spikes_1d_signal(tz_C)
    roll_C = filter_spikes_1d_signal(roll_C)
    pitch_C = filter_spikes_1d_signal(pitch_C)
    yaw_C = filter_spikes_1d_signal(yaw_C)
    tx_D = filter_spikes_1d_signal(tx_D)
    ty_D = filter_spikes_1d_signal(ty_D)
    tz_D = filter_spikes_1d_signal(tz_D)
    roll_D = filter_spikes_1d_signal(roll_D)
    pitch_D = filter_spikes_1d_signal(pitch_D)
    yaw_D = filter_spikes_1d_signal(yaw_D)
    

    # Plot X: TODO: need to fix the lengths if some tags are missing in some frames.
    # Also separate each tag frame idxs plots.
    ax_x.plot(tx_A, label=f'Tag {tagA}', color='r')
    ax_x.plot(tx_B, label=f'Tag {tagB}', color='g')
    ax_x.plot(tx_C, label=f'Tag {tagC}', color='b')
    ax_x.plot(tx_D, label=f'Tag {tagD}', color='y')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X (m) Displacement from its intial pose')
    ax_x.legend()
    ax_x.grid()

    # Plot Y
    ax_y.plot(ty_A, label=f'Tag {tagA}', color='r')
    ax_y.plot(ty_B, label=f'Tag {tagB}', color='g')
    ax_y.plot(ty_C, label=f'Tag {tagC}', color='b')
    ax_y.plot(ty_D, label=f'Tag {tagD}', color='y')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y (m) Displacement from its intial pose')
    ax_y.legend()
    ax_y.grid()

    # Plot Z
    ax_z.plot(tz_A, label=f'Tag {tagA}', color='r')
    ax_z.plot(tz_B, label=f'Tag {tagB}', color='g')
    ax_z.plot(tz_C, label=f'Tag {tagC}', color='b')
    ax_z.plot(tz_D, label=f'Tag {tagD}', color='y')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z (m) Displacement from its intial pose')
    ax_z.legend()
    ax_z.grid()

    # Plot Roll
    ax_roll.plot(roll_A, label=f'Tag {tagA}', color='r')
    ax_roll.plot(roll_B, label=f'Tag {tagB}', color='g')
    ax_roll.plot(roll_C, label=f'Tag {tagC}', color='b')
    ax_roll.plot(roll_D, label=f'Tag {tagD}', color='y')
    ax_roll.set_xlabel('Frame Index')
    ax_roll.set_ylabel('Roll (rad) Displacement from its intial pose')
    ax_roll.legend()
    ax_roll.grid()

    # Plot Pitch
    ax_pitch.plot(pitch_A, label=f'Tag {tagA}', color='r')
    ax_pitch.plot(pitch_B, label=f'Tag {tagB}', color='g')
    ax_pitch.plot(pitch_C, label=f'Tag {tagC}', color='b')
    ax_pitch.plot(pitch_D, label=f'Tag {tagD}', color='y')
    ax_pitch.set_xlabel('Frame Index')
    ax_pitch.set_ylabel('Pitch (rad) Displacement from its intial pose')
    ax_pitch.legend()
    ax_pitch.grid()

    # Plot Yaw
    ax_yaw.plot(yaw_A, label=f'Tag {tagA}', color='r')
    ax_yaw.plot(yaw_B, label=f'Tag {tagB}', color='g')
    ax_yaw.plot(yaw_C, label=f'Tag {tagC}', color='b')
    ax_yaw.plot(yaw_D, label=f'Tag {tagD}', color='y')
    ax_yaw.set_xlabel('Frame Index')
    ax_yaw.set_ylabel('Yaw (rad) Displacement from its intial pose')
    ax_yaw.legend()
    ax_yaw.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
 
'''
Plot the individual x, y, z, roll, pitch, yaw signals of multiple specific physical limit based filtered tags over time: Displcement from initial position.
'''
def plot_physical_filtered_tags_displacement(rel_tag_poses, ax_x, ax_y, ax_z, ax_roll, ax_pitch, ax_yaw, tagA=None, tagB=None, tagC=None, tagD=None):
    ax_x.clear()
    ax_y.clear()
    ax_z.clear()

    frame_idxs = sorted(rel_tag_poses.keys())

    tx_A, ty_A, tz_A, roll_A, pitch_A, yaw_A = [], [], [], [], [], []
    tx_B, ty_B, tz_B, roll_B, pitch_B, yaw_B = [], [], [], [], [], []
    tx_C, ty_C, tz_C, roll_C, pitch_C, yaw_C = [], [], [], [], [], []
    tx_D, ty_D, tz_D, roll_D, pitch_D, yaw_D = [], [], [], [], [], []

    default_value = 0

    # Calculate the starting positions for displacement calculation
    # Right now I am doing relative to the first detected pose.

    # TODO: Assuming the first frame has all tags visible: Necessary for this to work.
    start_pos = {}
    poses = rel_tag_poses[frame_idxs[0]]
    if tagA in poses:
        # pos_A, rot_A = poses[tagA]
        start_pos[tagA] = poses[tagA]
    if tagB in poses:
        # pos_B, _ = poses[tagB]
        start_pos[tagB] = poses[tagB]
    if tagC in poses:
        # pos_C, _ = poses[tagC]
        start_pos[tagC] = poses[tagC]
    if tagD in poses:
        # pos_D, _ = poses[tagD]
        start_pos[tagD] = poses[tagD]

    for frame_idx in frame_idxs:
        poses = rel_tag_poses[frame_idx]
        if tagA in poses:
            pos_A, quat_A = calculate_rel_pose(poses[tagA], start_pos[tagA])
            tx_A.append(pos_A[0])
            ty_A.append(pos_A[1])
            tz_A.append(pos_A[2])
            r = R.from_quat(quat_A)  
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_A.append(roll)
            pitch_A.append(pitch)
            yaw_A.append(yaw)
        else:
            tx_A.append(default_value)
            ty_A.append(default_value)
            tz_A.append(default_value)
            roll_A.append(default_value)
            pitch_A.append(default_value)
            yaw_A.append(default_value)
        if tagB in poses:
            pos_B, quat_B = calculate_rel_pose(poses[tagB], start_pos[tagB])
            tx_B.append(pos_B[0])
            ty_B.append(pos_B[1])
            tz_B.append(pos_B[2])
            r = R.from_quat(quat_B)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_B.append(roll)
            pitch_B.append(pitch)
            yaw_B.append(yaw)

        else:
            tx_B.append(default_value)
            ty_B.append(default_value)
            tz_B.append(default_value)
            roll_B.append(default_value)
            pitch_B.append(default_value)
            yaw_B.append(default_value)
        if tagC in poses:
            pos_C, quat_C = calculate_rel_pose(poses[tagC], start_pos[tagC])
            tx_C.append(pos_C[0])
            ty_C.append(pos_C[1])
            tz_C.append(pos_C[2])
            r = R.from_quat(quat_C)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_C.append(roll)
            pitch_C.append(pitch)
            yaw_C.append(yaw)
        else:
            tx_C.append(default_value)
            ty_C.append(default_value)
            tz_C.append(default_value)
            roll_C.append(default_value)
            pitch_C.append(default_value)
            yaw_C.append(default_value)
        if tagD in poses:
            pos_D, quat_D = calculate_rel_pose(poses[tagD], start_pos[tagD])
            tx_D.append(pos_D[0])
            ty_D.append(pos_D[1])
            tz_D.append(pos_D[2])
            r = R.from_quat(quat_D)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            roll_D.append(roll)
            pitch_D.append(pitch)
            yaw_D.append(yaw)
        else:   
            tx_D.append(default_value)
            ty_D.append(default_value)
            tz_D.append(default_value)
            roll_D.append(default_value)
            pitch_D.append(default_value)
            yaw_D.append(default_value)

    # Filtering by removing physical impossible points. 
    tx_A = physical_signal_filter_translation(tx_A)
    ty_A = physical_signal_filter_translation(ty_A)
    tz_A = physical_signal_filter_translation(tz_A)
    tx_B = physical_signal_filter_translation(tx_B)
    ty_B = physical_signal_filter_translation(ty_B)
    tz_B = physical_signal_filter_translation(tz_B)
    tx_C = physical_signal_filter_translation(tx_C)
    ty_C = physical_signal_filter_translation(ty_C)
    tz_C = physical_signal_filter_translation(tz_C)
    tx_D = physical_signal_filter_translation(tx_D)
    ty_D = physical_signal_filter_translation(ty_D)
    tz_D = physical_signal_filter_translation(tz_D)

    roll_A = physical_signal_filter_rotation(roll_A)
    pitch_A = physical_signal_filter_rotation(pitch_A)
    yaw_A = physical_signal_filter_rotation(yaw_A)
    roll_B = physical_signal_filter_rotation(roll_B)
    pitch_B = physical_signal_filter_rotation(pitch_B)
    yaw_B = physical_signal_filter_rotation(yaw_B)
    roll_C = physical_signal_filter_rotation(roll_C)
    pitch_C = physical_signal_filter_rotation(pitch_C)
    yaw_C = physical_signal_filter_rotation(yaw_C)
    roll_D = physical_signal_filter_rotation(roll_D)
    pitch_D = physical_signal_filter_rotation(pitch_D)
    yaw_D = physical_signal_filter_rotation(yaw_D)
    

    # Plot X: TODO: need to fix the lengths if some tags are missing in some frames.
    # Also separate each tag frame idxs plots.
    ax_x.plot(tx_A, label=f'Tag {tagA}', color='r')
    ax_x.plot(tx_B, label=f'Tag {tagB}', color='g')
    ax_x.plot(tx_C, label=f'Tag {tagC}', color='b')
    ax_x.plot(tx_D, label=f'Tag {tagD}', color='y')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X (m) Displacement from its intial pose')
    ax_x.legend()
    ax_x.grid()

    # Plot Y
    ax_y.plot(ty_A, label=f'Tag {tagA}', color='r')
    ax_y.plot(ty_B, label=f'Tag {tagB}', color='g')
    ax_y.plot(ty_C, label=f'Tag {tagC}', color='b')
    ax_y.plot(ty_D, label=f'Tag {tagD}', color='y')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y (m) Displacement from its intial pose')
    ax_y.legend()
    ax_y.grid()

    # Plot Z
    ax_z.plot(tz_A, label=f'Tag {tagA}', color='r')
    ax_z.plot(tz_B, label=f'Tag {tagB}', color='g')
    ax_z.plot(tz_C, label=f'Tag {tagC}', color='b')
    ax_z.plot(tz_D, label=f'Tag {tagD}', color='y')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z (m) Displacement from its intial pose')
    ax_z.legend()
    ax_z.grid()

    # Plot Roll
    ax_roll.plot(roll_A, label=f'Tag {tagA}', color='r')
    ax_roll.plot(roll_B, label=f'Tag {tagB}', color='g')
    ax_roll.plot(roll_C, label=f'Tag {tagC}', color='b')
    ax_roll.plot(roll_D, label=f'Tag {tagD}', color='y')
    ax_roll.set_xlabel('Frame Index')
    ax_roll.set_ylabel('Roll (rad) Displacement from its intial pose')
    ax_roll.legend()
    ax_roll.grid()

    # Plot Pitch
    ax_pitch.plot(pitch_A, label=f'Tag {tagA}', color='r')
    ax_pitch.plot(pitch_B, label=f'Tag {tagB}', color='g')
    ax_pitch.plot(pitch_C, label=f'Tag {tagC}', color='b')
    ax_pitch.plot(pitch_D, label=f'Tag {tagD}', color='y')
    ax_pitch.set_xlabel('Frame Index')
    ax_pitch.set_ylabel('Pitch (rad) Displacement from its intial pose')
    ax_pitch.legend()
    ax_pitch.grid()

    # Plot Yaw
    ax_yaw.plot(yaw_A, label=f'Tag {tagA}', color='r')
    ax_yaw.plot(yaw_B, label=f'Tag {tagB}', color='g')
    ax_yaw.plot(yaw_C, label=f'Tag {tagC}', color='b')
    ax_yaw.plot(yaw_D, label=f'Tag {tagD}', color='y')
    ax_yaw.set_xlabel('Frame Index')
    ax_yaw.set_ylabel('Yaw (rad) Displacement from its intial pose')
    ax_yaw.legend()
    ax_yaw.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

#######################################################################################
#                  Plot the Fused Centroid Path of External Tag
#######################################################################################

# Defining rigid transformation matrix from each id to centroid [ Assuming facce 1 as front]
# T_1C: 1 wrt Centroid (0,0,0)
# T_C1: Centroid (0,0,0) wrt 1
T_1C = np.array([[1, 0, 0, cube_size / 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

T_C1 = np.array([[1, 0, 0, -cube_size / 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

T_2C = np.array([[0, 0, 1, 0],
                [0, 1, 0, cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C2 = np.array([[0, 0, 1, 0],
                [0, 1, 0, -cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_3C = np.array([[0, 0, 1, -cube_size / 2],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C3 = np.array([[0, 0, 1, cube_size / 2],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_4C = np.array([[0, 0, 1, 0],
                [0, 1, 0, -cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C4 = np.array([[0, 0, 1, 0],
                [0, 1, 0, cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_5C = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, cube_size / 2],
                [0, 0, 0, 1]])

T_C5 = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, - cube_size / 2],
                [0, 0, 0, 1]])

# Global variable to store fused centroid path.
fused_centroid_path = [] 
total_frames = 0

# Funtion to read tag_paths_ext.csv to get current frame poses.
def read_frame_poses():
    all_frame_poses = []
    with open('frames_output/tag_paths_ext.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            pose = (
                int(row['frame_id']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )

            all_frame_poses.append((tag_id, pose[0] , pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7]))

    # print("Read the Current Frame Poses from csv with length of: ", len(all_frame_poses))
    return all_frame_poses

def get_current_frame_poses(all_frame_poses, frame_idx):
    current_frame_poses = [p for p in all_frame_poses if p[1] == frame_idx]   ###  is 0 or 1
    return current_frame_poses

def pose_fusion(poses):  #frame_idx might be needed to be removed evrywhere in this function.
    """
    Function to fuse multiple tag poses to get a more stable pose estimate for the centroid: based on the tags visible.
    Input: list of centroid poses from all visible tags [(frame_idx, tx, ty, tz, qx, qy, qz, qw), ...]
    Output: fused pose (tx, ty, tz, qx, qy, qz, qw)
    """

    if not poses:
        return None

    # frame_idx = poses[0][0]
    translations = np.array([[p[0], p[1], p[2]] for p in poses])
    quaternions = np.array([[p[3], p[4], p[5], p[6]] for p in poses])

    # Average translation
    avg_translation = np.mean(translations, axis=0)

    # Average quaternion (using Singular Value Decomposition method): 
    # Its a least squares solution to find the average quaternion.
    # More robust than simple averaging.
    A = np.zeros((4, 4))
    for q in quaternions:
        A += np.outer(q, q)
    A /= len(quaternions)
    _, _, Vt = np.linalg.svd(A)
    avg_quaternion = Vt[0]

    return ( avg_translation[0], avg_translation[1], avg_translation[2],
            avg_quaternion[0], avg_quaternion[1], avg_quaternion[2], avg_quaternion[3])

# TODO: Can be later merged with plot_relative_pose_indv.
'''
Plot the 3D relative pose of a specific external tag with respect to a static tag (ID = static_tag_id).

TODO: this has logic error, for cases when the mentioned tag is not visible in some frames. 
''' 
def plot_relative_pose_indvExt(tag_id, ax_indv, ay_indv, az_indv, all_frame_poses):
    ax_indv.clear()  # Clear the previous plot
    ay_indv.clear()
    az_indv.clear()
    
    static_tag_id = 0 # Ext tag static id.
    tag_paths = {}
    tx_rel = []
    ty_rel = []
    tz_rel = []

    total_frames = max(p[1] for p in all_frame_poses) + 1  # Assuming frame indices start from 0

    for frame_idx in range(total_frames):
        # Collecting same frame index poses, all tags.
        current_frame_poses = get_current_frame_poses(all_frame_poses, frame_idx)

        # check if any of the poses in current_frame_poses matches static_tag_id or tag_id
        if not any(int(p[0]) == static_tag_id for p in current_frame_poses) or not any(int(p[0]) == tag_id for p in current_frame_poses):
            # Static tag of Indv tag not visible in frame. 
            # print(f"Static Tag or Tag {tag_id} not found in frame {frame_idx}.")
            tx_rel.append(0)
            ty_rel.append(0)
            tz_rel.append(0)
            continue
        
        else:

            ref_pose = next((p for p in current_frame_poses if int(p[0]) == static_tag_id), None)
            p = next((p for p in current_frame_poses if int(p[0]) == tag_id), None)

            # Extract translation poses
            t_stat = np.array([p[2], p[3], p[4]])
            t_ref = np.array([ref_pose[2], ref_pose[3], ref_pose[4]])
            # Extract rotation poses
            R_stat = R.from_quat(p[5:9]).as_matrix()
            R_ref = R.from_quat(ref_pose[5:9]).as_matrix()

            # Compute relative transformation
            R_rel =  R_ref.T @ R_stat
            t_rel = R_ref.T @ (t_stat - t_ref)

            tx_rel.append(t_rel[0])
            ty_rel.append(t_rel[1])
            tz_rel.append(t_rel[2])

    # Calculate displacement from initial position  of tx_rel, ty_rel, tz_rel
    initial_tx = tx_rel[0]
    initial_ty = ty_rel[0]
    initial_tz = tz_rel[0]

    tx_rel = [x - initial_tx for x in tx_rel]
    ty_rel = [y - initial_ty for y in ty_rel]
    tz_rel = [z - initial_tz for z in tz_rel]


    ax_indv.plot(tx_rel, label=f'Tag {tag_id}')
    ay_indv.plot(ty_rel, label=f'Tag {tag_id}')
    az_indv.plot(tz_rel, label=f'Tag {tag_id}')
    ax_indv.set_xlabel('Frame Index')
    ax_indv.set_ylabel('Displacement X (m)')
    ay_indv.set_xlabel('Frame Index')
    ay_indv.set_ylabel('Displacement Y (m)')
    az_indv.set_xlabel('Frame Index')
    az_indv.set_ylabel('Displacement Z (m)')
    ax_indv.legend()
    ay_indv.legend()
    az_indv.legend()
    ax_indv.set_title(f'Displacement X Path of Tag {tag_id} wrt its initial pose')
    ay_indv.set_title(f'Displacement Y Path of Tag {tag_id} wrt its initial pose')
    az_indv.set_title(f'Displacement Z Path of Tag {tag_id} wrt its initial pose')

    # plt.show()

def calculate_centroid_path(all_frame_poses, start_frame=0, end_frame=None):
    # Centroid poses from all tags
    centroid_poses = []
    fused_centroid_path = []
    tx_rel = []
    ty_rel = []
    tz_rel = []

    total_frames = max(p[1] for p in all_frame_poses) + 1  # Assuming frame indices start from 0
    start_frame = 0
    if (end_frame is None):
        end_frame = total_frames

    for frame_idx in range(start_frame, end_frame):
        # Collecting same frame index poses, all tags.
        current_frame_poses = get_current_frame_poses(all_frame_poses, frame_idx)
        
        # Iterate through all tag paths and plot their centroid paths
        for pose in current_frame_poses:
            centroids = []
            # for pose in poses:
            tag_id = pose[0]
            _, frame_idx, tx, ty, tz, qx, qy, qz, qw = pose
            
            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([[rotation_matrix[0,0], rotation_matrix[0,1], rotation_matrix[0,2], tx],
                            [rotation_matrix[1,0], rotation_matrix[1,1], rotation_matrix[1,2], ty],
                            [rotation_matrix[2,0], rotation_matrix[2,1], rotation_matrix[2,2], tz],
                            [0, 0, 0, 1]])
            
            if tag_id == 1:
                T_centroid = T_C1 @ t
            elif tag_id == 2:
                T_centroid = T_C2 @ t
            elif tag_id == 3:
                T_centroid = T_C3 @ t
            elif tag_id == 4:
                T_centroid = T_C4 @ t
            elif tag_id == 5:
                T_centroid = T_C5 @ t
            elif tag_id == 0:
                static_tag_pose = pose            
            else:
                T_centroid = np.eye(4)

            if (tag_id != 0):
                # Extract translation and orientation
                tx, ty, tz = T_centroid[:3, 3]
                rotation_matrix = T_centroid[:3, :3]
                quaternion = R.from_matrix(rotation_matrix).as_quat()  # (qx, qy, qz, qw)
                qx, qy, qz, qw = quaternion
                centroid = [tx, ty, tz, qx, qy, qz, qw]
                centroid_poses.append(centroid)

        # Fuse the centroid poses from all visible tags to get a more stable estimate.
        fused_pose = pose_fusion(centroid_poses)

        # DEBUG: not fuse but take first centroid pose.
        # fused_pose = [tx, ty, tz, qx, qy, qz, qw] 

        # TODO: No visible tags case: if fused_pose is None: currently returning None, should handle it better.
        if fused_pose is not None:
            relative_fused_pose = []
            # relative_fused_pose.append(frame_idx) ##################################################### BEWARE : TODO ##########

            # Relative to static tag logic to be added here.
            # static_tag_pose --> (frame_idx, tx, ty, tz, qx, qy, qz, qw)
            # fused_pose --> (tx, ty, tz, qx, qy, qz, qw)
            # translation relative to static tag
            relative_fused_pose.append(fused_pose[0] - static_tag_pose[2])
            relative_fused_pose.append(fused_pose[1] - static_tag_pose[3])
            relative_fused_pose.append(fused_pose[2] - static_tag_pose[4])
            # rotation relative to static tag
            R_fused = R.from_quat(fused_pose[3:7]).as_matrix()
            R_static = R.from_quat(static_tag_pose[5:9]).as_matrix()
            R_rel = R_static.T @ R_fused
            quaternion_rel = R.from_matrix(R_rel).as_quat()
            relative_fused_pose.append(quaternion_rel[0])
            relative_fused_pose.append(quaternion_rel[1])
            relative_fused_pose.append(quaternion_rel[2])
            relative_fused_pose.append(quaternion_rel[3])

            fused_centroid_path.append(relative_fused_pose)
    
    return fused_centroid_path

'''
Plot the 3D centroid path of the cube relative to the static tag ID = 0 over all frames.
'''
def plot_centroid_path(ax_centroid, all_frame_poses, start_frame=0, end_frame=None, axis_equal=False):   
    # Clear the plot
    ax_centroid.clear()

    # Centroid poses from all tags
    centroid_poses = []
    fused_centroid_path = []
    tx_rel = []
    ty_rel = []
    tz_rel = []

    total_frames = max(p[0] for p in all_frame_poses) + 1  # Assuming frame indices start from 0
    if (end_frame is None):
        end_frame = total_frames
    
    print("Total Frames in Data: ", total_frames)
    print(f"Plotting Centroid Path from frame {start_frame} to {end_frame}")

    for frame_idx in range(start_frame, end_frame):
        # Collecting same frame index poses, all tags.
        current_frame_poses = get_current_frame_poses(all_frame_poses, frame_idx)

        
        # Iterate through all tag paths and plot their centroid paths
        for pose in current_frame_poses:
            centroids = []
            # for pose in poses:
            tag_id = pose[0]
            _, _, tx, ty, tz, qx, qy, qz, qw = pose
            t = np.array([[1, 0, 0, tx],
                            [0, 1, 0, ty],
                            [0, 0, 1, tz],
                            [0, 0, 0, 1]])
            if tag_id == 1:
                T_centroid = T_C1 @ t
            elif tag_id == 2:
                T_centroid = T_C2 @ t
            elif tag_id == 3:
                T_centroid = T_C3 @ t
            elif tag_id == 4:
                T_centroid = T_C4 @ t
            elif tag_id == 5:
                T_centroid = T_C5 @ t
            elif tag_id == 0:
                static_tag_pose = pose            
            else:
                T_centroid = np.eye(4)

            if (tag_id != 0):
                # Extract translation and orientation
                tx, ty, tz = T_centroid[:3, 3]
                rotation_matrix = T_centroid[:3, :3]
                quaternion = R.from_matrix(rotation_matrix).as_quat()  # (qx, qy, qz, qw)
                qx, qy, qz, qw = quaternion
                centroid = [tx, ty, tz, qx, qy, qz, qw]
                centroid_poses.append(centroid)

        # Fuse the centroid poses from all visible tags to get a more stable estimate.
        fused_pose = pose_fusion(centroid_poses)

        # TODO: No visible tags case: currently returning None, should handle it better.
        if fused_pose is not None:
            relative_fused_pose = []

            # Relative to static tag logic to be added here.
            # static_tag_pose --> (frame_idx, tx, ty, tz, qx, qy, qz, qw)
            # fused_pose --> (tx, ty, tz, qx, qy, qz, qw)
            # translation relative to static tag
            relative_fused_pose.append(fused_pose[0] - static_tag_pose[1])
            relative_fused_pose.append(fused_pose[1] - static_tag_pose[2])
            relative_fused_pose.append(fused_pose[2] - static_tag_pose[3])
            # rotation relative to static tag
            R_fused = R.from_quat(fused_pose[3:7]).as_matrix()
            R_static = R.from_quat(static_tag_pose[5:9]).as_matrix()
            R_rel = R_static.T @ R_fused
            quaternion_rel = R.from_matrix(R_rel).as_quat()
            relative_fused_pose.append(quaternion_rel[0])
            relative_fused_pose.append(quaternion_rel[1])
            relative_fused_pose.append(quaternion_rel[2])
            relative_fused_pose.append(quaternion_rel[3])

            fused_centroid_path.append(relative_fused_pose)

    # if fused_pose is None: 
    # TODO: change the logic, to have empty or something printed out not just return. no visible tags
    #     return
    # fused_centroid_path = np.array(fused_centroid_path)

    if len(fused_centroid_path) > 0:
        ax_centroid.plot(np.array(fused_centroid_path)[:, 0], np.array(fused_centroid_path)[:, 1], np.array(fused_centroid_path)[:, 2], c='r')
    else:
        pass # or use fused_centroid_path = [[0,0,0]] to plot a point at origin.

    # Set labels and title
    # I want all x,y and z range to be same for better visualization: ans use the largest range among them.
    
    ax_centroid.set_xlabel('X (m)')
    ax_centroid.set_ylabel('Y (m)')
    ax_centroid.set_zlabel('Z (m)')

    if axis_equal:
        ax_centroid.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

        # --- Make equal scale automatically ---
        x = np.array(fused_centroid_path)[:, 0]
        y = np.array(fused_centroid_path)[:, 1]
        z = np.array(fused_centroid_path)[:, 2]
        max_range = np.ptp([x, y, z]).max() / 2.0
        mid_x = (np.max(x) + np.min(x)) / 2.0
        mid_y = (np.max(y) + np.min(y)) / 2.0
        mid_z = (np.max(z) + np.min(z)) / 2.0

        ax_centroid.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_centroid.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_centroid.set_zlim(mid_z - max_range, mid_z + max_range)

    ax_centroid.set_title('Cube Centroid Path')
    ax_centroid.legend()
    plt.show()




'''
Plot all external tag paths in 3D space.
'''
def plot_all_ext_tags(ax_all, all_frame_poses):
    ax_all.clear()  # Clear the previous plot
    tag_paths = {}
    for row in all_frame_poses:
        tag_id = int(row[0]) #tag_id
        pose = (
            int(row[1]), #frame_id
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
            float(row[8])
        )
        tag_paths.setdefault(tag_id, []).append(pose)
    # Plot each tag's path
    for tag_id, poses in tag_paths.items():
        poses = sorted(poses, key=lambda x: x[0])
        tx = [p[1] for p in poses]
        ty = [p[2] for p in poses]
        tz = [p[3] for p in poses]
        ax_all.plot(tx, ty, tz, label=f'Tag {tag_id}')
    ax_all.set_xlabel('X (m)')
    ax_all.set_ylabel('Y (m)')
    ax_all.set_zlabel('Z (m)')
    ax_all.legend()
    ax_all.set_title('All External Tag Paths')
    plt.show()

def calculate_displacement(centroid_path):
    # displacement_centroid_path = {}
    displacement_centroid_path = []

    initial_pose = centroid_path[0]

    for pose in centroid_path:
        # frame_idx, tx, ty, tz, qx, qy, qz, qw = pose          /////////
        tx, ty, tz, qx, qy, qz, qw = pose

        t = np.array([[1, 0, 0, tx],
                        [0, 1, 0, ty],
                        [0, 0, 1, tz],
                        [0, 0, 0, 1]])
        rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
        # Calculate displacement from initial pose
        disp_tx = tx - initial_pose[0]
        disp_ty = ty - initial_pose[1]
        disp_tz = tz - initial_pose[2]
        # Calculate relative rotation
        R_curr = rotation_matrix
        R_init = R.from_quat([initial_pose[3], initial_pose[4], initial_pose[5], initial_pose[6]]).as_matrix()
        R_rel = R_init.T @ R_curr
        quaternion_rel = R.from_matrix(R_rel).as_quat()
        disp_qx, disp_qy, disp_qz, disp_qw = quaternion_rel
        displacement_centroid_path.append([disp_tx, disp_ty, disp_tz, disp_qx, disp_qy, disp_qz, disp_qw])
        # if frame_idx not in displacement_centroid_path:
        #     displacement_centroid_path[frame_idx] = {}
        # displacement_centroid_path[frame_idx] = (disp_tx, disp_ty, disp_tz, disp_qx, disp_qy, disp_qz, disp_qw)


    return displacement_centroid_path

#######################################################################################
#                     Syncing the Internal and External Time Stamps
#######################################################################################

def sync_frames(int_frame_list, ext_frame_list):
    """
    Realsense in at a fps of 30, while blue-os is at 10 fps (Trying to improve this but still wont be same)
    Function to sync internal and external frames based on nearest neighbor matching of timestamps.
    
    Input: 
        int_timestamps: list of internal timestamps timestamps_int.csv
        ext_timestamps: list of external timestamps from timestamps_ext.csv

    Output:
        synced_indices: mapping of frame indices.
    """
    
    # Use Int timestamp as base (i.e., for each Int frame, find the Ext frame with nearest timestamp).
    int_times = []
    int_valid_ids = set(int_frame_list)
    ext_valid_ids = set(ext_frame_list)
    int_ids = []
    int_times = []

    with open('frames_output/timestamps_int.csv', 'r') as f: 
        reader = csv.DictReader(f)
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row['frame_id'])
            if fid in int_valid_ids:                
                int_ids.append(fid)
                int_times.append(float(row['timestamp_sec']))

    ext_ids = []
    ext_times = []
    # i need to append only those ids that are present in ext_frame_list
    with open('frames_output/timestamps_ext.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row['frame_id'])
            if fid in ext_valid_ids:                
                ext_ids.append(fid)
                ext_times.append(float(row['timestamp_sec']))
    
    # Use bisect for nearest neighbor.
    mapping = []
    for i_id, i_time in zip(int_ids, int_times):
        # Find insertion point in ext_times
        idx = bisect.bisect_left(ext_times, i_time)

        # Check neighbors to find closest..
        if idx == 0:
            closest_idx = 0
        elif idx == len(ext_times):
            closest_idx = len(ext_times) - 1
        else:
            before = ext_times[idx - 1]
            after = ext_times[idx]
            if abs(i_time - before) <= abs(i_time - after):
                closest_idx = idx - 1
            else:
                closest_idx = idx

        mapping.append((i_id, ext_ids[closest_idx]))

    return mapping

def align_and_combine_data(rel_tag_poses, displacement_centroid_path, frame_idx_map):
    
    # TODO : Add checks to remove frames where either internal or external data is missing.   
    combined_data = []
    for frame_id_int, frame_id_ext in frame_idx_map:
        combined_data.append([frame_id_int, rel_tag_poses[frame_id_int], displacement_centroid_path[frame_id_ext]])
    
    # How combined data looks like: 
    # [1, {5: (...), 6: (...), 7: (...), 8: (...), 9: (...), 13: (...), 14: (...), 16: (...), 17: (...), 18: (...)}, (0.863955349068875, -1.0197836232984943, -5.915013798198963, 0.9456898669487789, 0.08426047535714518, -0.2504715065584917, 0.18930100962569735)]
    return combined_data
   
#######################################################################################
#                       Plotting Int and Ext Signals Combined
#######################################################################################

def plot_combined(centroid_path, rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=None, tagB=None, tagC = None, tagD=None):
    
    # # Clear all axes
    # centroid_x.clear()
    # centroid_y.clear()
    # centroid_z.clear()
    ax_x1.clear()
    ax_y1.clear()
    ax_z1.clear()
    ax_roll1.clear()
    ax_pitch1.clear()
    ax_yaw1.clear()

    # # Plot centroid path signals
    # if len(centroid_path) > 0:
    #     centroid_x.plot(np.array(centroid_path)[:, 0], label='Centroid X', color='c')
    #     centroid_y.plot(np.array(centroid_path)[:, 1], label='Centroid Y', color='m')
    #     centroid_z.plot(np.array(centroid_path)[:, 2], label='Centroid Z', color='y')

    # centroid_x.set_xlabel('Frame Index')
    # centroid_x.set_ylabel('Centroid X Position (m)')
    # centroid_x.legend()
    # centroid_x.grid()

    # centroid_y.set_xlabel('Frame Index')
    # centroid_y.set_ylabel('Centroid Y Position (m)')
    # centroid_y.legend()
    # centroid_y.grid()

    # centroid_z.set_xlabel('Frame Index')
    # centroid_z.set_ylabel('Centroid Z Position (m)')
    # centroid_z.legend()
    # centroid_z.grid()

    # Plot internal tag signals
    plot_unfiltered_tags_displacement(rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=tagA, tagB=tagB, tagC=tagC, tagD=tagD)

def plot_filtered_combined(centroid_path, rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=None, tagB=None, tagC = None, tagD=None):
    
    # # Clear all axes
    # centroid_x.clear()
    # centroid_y.clear()
    # centroid_z.clear()
    ax_x1.clear()
    ax_y1.clear()
    ax_z1.clear()
    ax_roll1.clear()
    ax_pitch1.clear()
    ax_yaw1.clear()

    # # Plot centroid path signals
    # if len(centroid_path) > 0:
    #     centroid_x.plot(np.array(centroid_path)[:, 0], label='Centroid X', color='c')
    #     centroid_y.plot(np.array(centroid_path)[:, 1], label='Centroid Y', color='m')
    #     centroid_z.plot(np.array(centroid_path)[:, 2], label='Centroid Z', color='y')

    # centroid_x.set_xlabel('Frame Index')
    # centroid_x.set_ylabel('Centroid X Position (m)')
    # centroid_x.legend()
    # centroid_x.grid()

    # centroid_y.set_xlabel('Frame Index')
    # centroid_y.set_ylabel('Centroid Y Position (m)')
    # centroid_y.legend()
    # centroid_y.grid()

    # centroid_z.set_xlabel('Frame Index')
    # centroid_z.set_ylabel('Centroid Z Position (m)')
    # centroid_z.legend()
    # centroid_z.grid()

    # Plot internal tag signals
    plot_physical_filtered_tags_displacement(rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=tagA, tagB=tagB, tagC=tagC, tagD=tagD)


def plot_velocity_filtered_combined(centroid_path, rel_tag_poses, centroid_x, centroid_y, centroid_z, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=None, tagB=None, tagC = None, tagD=None):
    
    # Clear all axes
    centroid_x.clear()
    centroid_y.clear()
    centroid_z.clear()
    ax_x1.clear()
    ax_y1.clear()
    ax_z1.clear()
    ax_roll1.clear()
    ax_pitch1.clear()
    ax_yaw1.clear()

    # Calculate velocity and acceleration for centroid path
    if len(centroid_path) > 1:
        centroid_path = np.array(centroid_path)
        velocity = np.diff(centroid_path[:, :3], axis=0)  # First derivative (velocity)
        acceleration = np.diff(velocity, axis=0)  # Second derivative (acceleration)

        # # Plot velocity signals
        # centroid_x.plot(velocity[:, 0], label='Centroid Velocity X', color='c', linestyle='--')
        # centroid_y.plot(velocity[:, 1], label='Centroid Velocity Y', color='m', linestyle='--')
        # centroid_z.plot(velocity[:, 2], label='Centroid Velocity Z', color='y', linestyle='--')

        # Plot acceleration signals
        centroid_x.plot(acceleration[:, 0], label='Centroid Acceleration X', color='c', linestyle=':')
        centroid_y.plot(acceleration[:, 1], label='Centroid Acceleration Y', color='m', linestyle=':')
        centroid_z.plot(acceleration[:, 2], label='Centroid Acceleration Z', color='y', linestyle=':')

    # Plot centroid path signals
    centroid_x.set_xlabel('Frame Index')
    centroid_x.set_ylabel('Centroid X Acceleration (m)')
    centroid_x.legend()
    centroid_x.grid()

    centroid_y.set_xlabel('Frame Index')
    centroid_y.set_ylabel('Centroid Y Acceleration (m)')
    centroid_y.legend()
    centroid_y.grid()

    centroid_z.set_xlabel('Frame Index')
    centroid_z.set_ylabel('Centroid Z Acceleration (m)')
    centroid_z.legend()
    centroid_z.grid()

    # Plot internal tag signals
    plot_physical_filtered_tags_displacement(rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=tagA, tagB=tagB, tagC=tagC, tagD=tagD)


#######################################################################################
#                           Telemetry Data Plot
#######################################################################################



#######################################################################################
#                           Perception Model
#######################################################################################

####################################################
#       Perception Model: Wavelet Transform
####################################################

def perception_wavelet_model_predict(filtered_signals):
    """
    Function to predict the perception model based on statistical analysis of filtered signals.
    Input: filtered_signals is dictionary of [tag_id] [x,y,z,roll,pitch,yaw] : signal vs time(frame index)
    Output: perception model prediction
    """

    # Convert the signal in time domain to frequency domain.

    # Convert the signal in time domain to frequency domain using FFT
    frequency_signals = {}
    for tag_id, signals in filtered_signals.items():
        frequency_signals[tag_id] = {}
        for axis, signal in signals.items():
            frequency_signals[tag_id][axis] = np.fft.fft(signal)

    # Wavelet Analysis: Haar 
    # Perform Wavelet Transform on the frequency signals
    wavelet_signals = {}
    for tag_id, signals in frequency_signals.items():
        wavelet_signals[tag_id] = {}
        for axis, freq_signal in signals.items():
            # Using Haar wavelet for decomposition
            coeffs = pywt.wavedec(np.abs(freq_signal), 'haar', level=4)
            wavelet_signals[tag_id][axis] = coeffs
    
    # Analysis
    # Perform statistical analysis on wavelet coefficients
    perception = {}
    for tag_id, signals in wavelet_signals.items():
        perception[tag_id] = {}
        for axis, coeffs in signals.items():
            # Calculate statistical features like mean, variance, skewness, kurtosis
            mean_coeff = np.mean(coeffs[0])  # Approximation coefficients
            var_coeff = np.var(coeffs[0])
            skew_coeff = scipy.stats.skew(coeffs[0])
            kurt_coeff = scipy.stats.kurtosis(coeffs[0])
            
            # Store the features for the axis
            perception[tag_id][axis] = {
                "mean": mean_coeff,
                "variance": var_coeff,
                "skewness": skew_coeff,
                "kurtosis": kurt_coeff
            }
            # Perform classification based on statistical features
            # Example: Classify based on mean and variance thresholds
            classification = {}
            for tag_id, axes in perception.items():
                classification[tag_id] = {}
                for axis, stats in axes.items():
                    if stats["mean"] > 0.5 and stats["variance"] < 0.1:
                        classification[tag_id][axis] = "Stable"
                    elif stats["mean"] < -0.5 and stats["variance"] < 0.1:
                        classification[tag_id][axis] = "Unstable"
                    else:
                        classification[tag_id][axis] = "Unknown"
            
        return classification

def plot_wavelet_transformed_signal(filtered_signals, tag_id, axis):
        """
        Function to plot the wavelet transformed signal for a specific tag and axis.
        Input: 
            filtered_signals: dictionary of [tag_id] [x,y,z,roll,pitch,yaw] : signal vs time(frame index)
            tag_id: the tag ID to plot
            axis: the axis to plot (e.g., 'x', 'y', 'z', 'roll', 'pitch', 'yaw')
        """
        if tag_id not in filtered_signals or axis not in filtered_signals[tag_id]:
            print(f"Tag {tag_id} or axis {axis} not found in filtered_signals.")
            return

        signal = filtered_signals[tag_id][axis]

        coeffs = pywt.wavedec(signal, 'haar', level=1)
        reconstructed_signals = pywt.waverec(coeffs, 'haar')

        # Plot original and the wavelet-transformed reconstructed signals.
        plt.figure(figsize=(12, 6))
        plt.plot(signal, label='Original Signal', color='blue')
        plt.plot(reconstructed_signals, label='Wavelet Transformed Signal', color='orange', linestyle='--')
        plt.title(f"Wavelet Transform of Tag {tag_id} - {axis.upper()} Axis")
        plt.xlabel("Frame Index")
        plt.ylabel("Signal Value")
        plt.legend()
        plt.grid()
        plt.show()

def plot_wavelet_components(filtered_signals, tag_id, axis):
    """
    Plots individual wavelet components A4, D4, D3, D2, D1 for a given tag and axis.
    Input:
        filtered_signals: dict[tag_id][axis] => signal
        tag_id: int or str
        axis: 'x','y','z','roll','pitch','yaw'
    """
    if tag_id not in filtered_signals or axis not in filtered_signals[tag_id]:
        print(f"Tag {tag_id} or axis {axis} not found in filtered_signals.")
        return

    signal = filtered_signals[tag_id][axis]
    coeffs = pywt.wavedec(signal, 'haar', level=6)  # if level 4: A4, D4, D3, D2, D1
    labels4 = ['A4 (Approximation)', 'D4 (Detail)', 'D3 (Detail)', 'D2 (Detail)', 'D1 (Detail)']
    labels6 = ['A6 (Approximation)','D6 (Detail)','D5 (Detail)', 'D4 (Detail)', 'D3 (Detail)', 'D2 (Detail)', 'D1 (Detail)']

    # plt.figure(figsize=(14, 10)) 4
    plt.figure(figsize=(15, 14))   # 6
    for i, comp in enumerate(coeffs):
        plt.subplot(7, 1, i+1)
        plt.plot(comp)
        plt.title(f"{labels6[i]} Component - Tag {tag_id} | Axis {axis.upper()}")
        plt.xlabel("Coefficient Index")
        plt.ylabel("Value")
        plt.grid()

    plt.tight_layout()
    plt.show()


def detect_u_shapes(filtered_signals, tag_id, axis, threshold=0.05, min_width=100):
    """
    Detects U-shape start points using Haar wavelet approximation (A6) and plots them.
    
    Args:
        filtered_signals: dict[tag_id][axis] => signal array
        tag_id: key for the required signal
        axis: 'x','y','z','roll','pitch','yaw'
        threshold: vertical tolerance around 0 line to accept a U-bottom
        min_width: minimum number of frames from start->bottom->rise to call it a U
    """

    if tag_id not in filtered_signals or axis not in filtered_signals[tag_id]:
        print("Tag or axis not found.")
        return

    signal = np.array(filtered_signals[tag_id][axis])

    # ---- 6-level Haar decomposition ----
    coeffs = pywt.wavedec(signal, 'haar', level=6)
    A6 = coeffs[0]                         # top-level approximation
    A6_up = pywt.upcoef('a', A6, 'haar', level=6, take=len(signal))  # Upsample A6 to full length

    # ---- U-shape detection ----
    u_starts = []
    descending = False
    start_idx = None

    for i in range(2, len(A6_up)):
        prev = A6_up[i-1]
        cur = A6_up[i]
        nxt = A6_up[i] if i == len(A6_up)-1 else A6_up[i+1]

        # Detect beginning of downward slope
        if not descending and cur < prev:
            descending = True
            start_idx = i-1

        # Detect bottom + upward slope => U-shape confirmed
        if descending and cur < threshold and nxt > cur:
            # Validate minimum width of U
            if start_idx is not None and (i - start_idx) >= min_width:
                u_starts.append(start_idx)
            descending = False
            start_idx = None

    # ---- Plotting ----
    plt.figure(figsize=(14, 6))
    plt.plot(signal, label='Original Signal', linewidth=1.2)
    plt.plot(A6_up, label='A6 Wavelet Approximation', linestyle='--')

    print("Printing the change indexes: ")
    for idx in u_starts:
        plt.axvline(x=idx, color='red', linestyle='--', alpha=0.8)
        plt.text(idx, signal[idx], 'U-start', color='red')
        print("idx: ", idx)

    plt.title(f"U-shape Detection using Wavelet Approximation (Tag {tag_id} | Axis {axis.upper()})")
    plt.xlabel("Frame Index")
    plt.ylabel("Signal Value")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return u_starts
    
def detect_u_shapes_partial_causal(filtered_signals, tag_id, axis, threshold=0.05, min_width=100):
    """
    Partial causal (real-time) U-shape detector.
    Uses only past -> current values. No future lookahead.
    Marks the start of the U when a downward trend hits a valid bottom.

    Input:
    threshold: numeric tolerance used to decide whether a bottom is “near zero”.
    """

    if tag_id not in filtered_signals or axis not in filtered_signals[tag_id]:
        print("Tag or axis not found.")
        return

    signal = np.array(filtered_signals[tag_id][axis])

    # ---- Compute A6 ----
    coeffs = pywt.wavedec(signal, 'haar', level=6)
    A6 = coeffs[0]
    # upsamples / reconstructs the approximation component to the original signal length so we can align A6 with time indices of signal.
    A6_up = pywt.upcoef('a', A6, 'haar', level=6, take=len(signal))

    # ---- Causal U detection ----
    u_starts = []
    descending = False
    start_idx = None

    prev = A6_up[0]

    for i in range(1, len(A6_up)):

        cur = A6_up[i]
        slope = cur - prev

        # --- Detect start of descent ---
        if not descending and slope < 0:
            descending = True
            start_idx = i - 1

        # --- Detect bottom: slope turns upward (causal!) ---
        if descending and slope > 0:
            bottom_idx = i - 1

            # Check if bottom is near zero & wide enough
            if abs(A6_up[bottom_idx]) < threshold:
                if start_idx is not None and (bottom_idx - start_idx) >= min_width:
                    u_starts.append(start_idx)

            # Reset after confirming bottom
            descending = False
            start_idx = None

        prev = cur

    # ---- Plot results ----
    plt.figure(figsize=(14, 6))
    plt.plot(signal, label='Original Signal', linewidth=1.2, color='orange')
    plt.plot(A6_up, label='A6 Wavelet Approximation', linestyle='--', color='blue')
    
    # Set background color to grey
    plt.gca().set_facecolor('lightgrey')

    # print("Printing the index: ")
    for idx in u_starts:
        plt.axvline(x=idx, color='green', linestyle='--', alpha=0.9, linewidth=2.1)
        plt.text(idx, signal[idx], ' U-start', color='green', fontsize=9, fontweight='bold')
        print("idx: ", idx)

    plt.title(f"Point of Intrest Detection (Tag {tag_id} | Axis {axis.upper()})")
    plt.xlabel("Frame Index")
    plt.ylabel("Individual Axis Signal")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return u_starts


#######################################################################################
#                           Execution Loop
#######################################################################################

if __name__ == "__main__":

    ###########################
            # Internal
    ###########################

    # # fig = plt.figure(figsize=(12, 6))
    # # ax3 = fig.add_subplot(111, projection='3d')

    # # ax4 = fig.add_subplot(234, projection='3d')
    # # ax5 = fig.add_subplot(235, projection='3d')

    # # Create subplots for x, y, z signals: for plot indv function.
    # # fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(8, 12))

    # fig, (ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1) = plt.subplots(6, 1, figsize=(8, 24))
    # fig.suptitle(f'Separate Axes of Tag Over Time')

    ###########################    
            # External
    ###########################

    # # # Creating the figure and axes once for cube centroid plotting.
    # fig_ext2 = plt.figure()
    # ax_centroid = fig_ext2.add_subplot(111, projection='3d')
    # fig_ext, (ax, ay, az) = plt.subplots(3, 1, figsize=(8, 18))

    ###########################
            # Combined
    ###########################
    
    # # fig, (ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, centroid_x, centroid_y, centroid_z) = plt.subplots(9, 1, figsize=(8, 36))
    # fig, (ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1) = plt.subplots(6, 1, figsize=(8, 24))
    # fig.suptitle(f'Separate Axes of Tags Over Time')

    # # fig, (bx_x1, bx_y1, bx_z1, bx_roll1, bx_pitch1, bx_yaw1, centroid2_x, centroid2_y, centroid2_z) = plt.subplots(9, 1, figsize=(8, 36))
    # fig, (bx_x1, bx_y1, bx_z1, bx_roll1, bx_pitch1, bx_yaw1) = plt.subplots(6, 1, figsize=(8, 24))
    # # fig, (centroid2_x, centroid2_y, centroid2_z) = plt.subplots(3, 1, figsize=(8, 12))
    # fig.suptitle(f'Separate Axes of Tags Over Time - Filtered Internal Paths')

    ###########################
            # Telemetry
    ###########################ax

    while True:

        ###########################
                # Internal
        ###########################
        
        # # rel_tag_poses = filter_false_positive()
        # rel_tag_poses = unfilter_rel_paths()
        # # plot_filtered_tags(rel_tag_poses, ax_x, ax_y, ax_z, tagA=5, tagB=0, tagC=14, tagD=5)
        # # plot_filtered_tags_displacement(rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=18, tagB =14, tagC = 17)

        # plot_filtered_tags_displacement(rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagA=9, tagB =8)

        # plot_idv(17, ax_x, ax_y, ax_z)                 # IMP ## x,y,z signals wrt time for a tag.

        # Unused: plot_idv_denoised( 19, 1, ax5, ax_x, ax_y, ax_z)  ## x,y,z signals wrt time for a tag denoised by static tag
        # plot_6d_pose(ax1)
        # plot_3d_relative_pose(ax2)
        # plot_relative_pose_indv(1, ax3) 

        ###########################
                # External
        ###########################

        # # # Read from csv to get the data in this form:
        # all_frame_poses = read_frame_poses()
        # # curr_tag_id = 5
        # # # plot_all_ext_tags(ax_centroid, all_frame_poses)
        # plot_centroid_path(ax_centroid, all_frame_poses, axis_equal=False)
        # # plot_centroid_path_indv(ax, ay, az, all_frame_poses axis_equal=False)
        # # plot_relative_pose_indvExt(5, ax, ay, az, all_frame_poses)


        ###########################
                # Combined
        ###########################

        # rel_tag_poses = unfilter_rel_paths()
        # all_frame_poses = read_frame_poses()
        # centroid_path = calculate_centroid_path(all_frame_poses)
        # displacement_centroid_path = calculate_displacement(centroid_path)
        # plot_combined(displacement_centroid_path, rel_tag_poses, ax_x1, ax_y1, ax_z1, ax_roll1, ax_pitch1, ax_yaw1, tagB=17, tagA=14)


        # rel_tag_poses2 = unfilter_rel_paths()
        # all_frame_poses2 = read_frame_poses()
        # centroid_path2 = calculate_centroid_path(all_frame_poses2)
        # displacement_centroid_path2 = calculate_displacement(centroid_path2)
        # plot_filtered_combined(displacement_centroid_path2, rel_tag_poses2, bx_x1, bx_y1, bx_z1, bx_roll1, bx_pitch1, bx_yaw1, tagB=17, tagA=14)


        # rel_tag_poses2 = unfilter_rel_paths()
        # all_frame_poses2 = read_frame_poses()
        # centroid_path2 = calculate_centroid_path(all_frame_poses2)
        # displacement_centroid_path2 = calculate_displacement(centroid_path2)
        # plot_velocity_filtered_combined(displacement_centroid_path2, rel_tag_poses2, centroid2_x, centroid2_y, centroid2_z, bx_x1, bx_y1, bx_z1, bx_roll1, bx_pitch1, bx_yaw1,tagB=17, tagC=14)

        ######################################################
                # Perception Model: Statistical
        ######################################################

        rel_tag_poses2 = unfilter_rel_paths()
        # filtered_signals is dictionary of [tag_id] [x,y,z,roll,pitch,yaw] : signal vs time(frame index)
        filtered_signals = get_filtered_signals(rel_tag_poses2, tagA=None, tagB=17, tagC=14, tagD=None)
        
        # Wavelet plot
        # plot_wavelet_transformed_signal(filtered_signals, tag_id=17, axis='z')
        # plot_wavelet_components(filtered_signals, tag_id=17, axis='z')
        # detect_u_shapes(filtered_signals, tag_id=17, axis='z')


        detect_u_shapes_partial_causal(filtered_signals, tag_id=17, axis='z')


        
        # # predict_label = perception_wavelet_model_predict(filtered_signals)



        




        ###########################
                # Telemetry
        ###########################

        plt.show()






# TODO: 
# 1. Add checks for stability values correponsiding to both int and ext and use frames only when both are stable.
# I am doing per frame classification and not memory buffer based regression or prediction, so this should be fine.
# 2. denoising/ comparision funtion for internal tags does not make a lot of sense right now.

# To Note IMP:  
# before usinf the ratation part for the static tag relaative transform, and after the plot remains the same : Wierd, considering 45 degree tilt not same plane...??
