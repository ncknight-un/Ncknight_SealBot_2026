import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import cv2


######################################################################
#                   Internal Camera Config
######################################################################


internal_tag_id_rel_pose_map = []
static_tag_id = 4
fxInt, fyInt, cxInt, cyInt = 1806.68775, 1801.14087, 1882.18218, 1404.07528   # With OpenCV Calib Matrix. TODO: verify std units here same as openCV.
tag_sizeInt = 0.007725     # The black square width ≈ 0.75 × 10.3 mm = 7.725 mm.   || 10.3 mm is with border
KInt = np.array([[fxInt, 0, cxInt],
                [0, fyInt, cyInt],
                [0,  0,  1]], dtype=np.float64)


# Calculate the relative pose of all the tags with respect to the static tag, for each frame in the CSV file
def calculate_relative_poses(csv_file):
    # Read the CSV file
    # Extract the rows with the same frame_id
    # For each frame, find the pose of the static tag
    # Calculate the relative pose of all other tags with respect to the static tag
    # Calculate the mean relative poses in internal_tag_id_rel_pose_map for all frames, for each tag.

    global internal_tag_id_rel_pose_map 
    tag_poses = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row['frame_idx'])
            tag_id = int(row['tag_id'])
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

    rel_poses_accum = {}

    for frame_id, poses in tag_poses.items():
        if static_tag_id not in poses: # this discards all frames where the static tag is not detected.
            continue

        static_pos, static_quaternion = poses[static_tag_id]
        static_rot = R.from_quat(static_quaternion)

        for tag_id, (pos, quat) in poses.items():
            if tag_id == static_tag_id:
                continue

            rel_pos = static_rot.inv().apply(pos - static_pos)
            rot = R.from_quat(quat)
            rel_rot = static_rot.inv() * rot

            if tag_id not in rel_poses_accum:
                rel_poses_accum[tag_id] = {'pos': [], 'rot': []}

            rel_poses_accum[tag_id]['pos'].append(rel_pos)
            rel_poses_accum[tag_id]['rot'].append(rel_rot)


    internal_tag_id_rel_pose_map = []
    for tag_id, data in rel_poses_accum.items():
        if (tag_id != 4 and tag_id <= 18 and tag_id != 15):  # Exclude static tag and tags > 18
            mean_pos = np.mean(data['pos'], axis=0)

            # Average quaternion (using Singular Value Decomposition method): 
            # Its a least squares solution to find the average quaternion.
            # More robust than simple averaging.
            quaternions = np.array([r.as_quat() for r in data['rot']])
            A = np.zeros((4, 4))
            for q in quaternions:
                A += np.outer(q, q)
            A /= len(quaternions)
            _, _, Vt = np.linalg.svd(A)
            mean_quaternion = Vt[0]

            internal_tag_id_rel_pose_map.append((tag_id, mean_pos, mean_quaternion))
    return internal_tag_id_rel_pose_map

group1 = [3,10,15,9,5,7,8,13,6]
group2 = [2,14,12,11,1,0,17,16,18]

def reProject_pose(rel_pose_map):
    # Take video file input, and remap the final pose of each tag to the frames in the video.
    # Output is the original video with the axis drawn on each tag, using the remapped pose.
    
    video_path = 'ref_data/tag_paths_ref_video.avi'
    cap = cv2.VideoCapture(video_path)
    tag_size = tag_sizeInt
    K = KInt
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_tag_reprojection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for tag_id, pos, quat in rel_pose_map:

            t = np.array(pos)
            r = R.from_quat(quat)
            R_mat = r.as_matrix()

            # 3D Reprojection of the detected tag pose.                        
            dist_coeffs = np.zeros(5)
            # Rodrigues rotation
            rvec, _ = cv2.Rodrigues(R_mat)
            tvec = t.reshape(3, 1)

            # --- 1) Project tag origin ---
            origin_3d = np.array([[0, 0, 0]], dtype=float)
            origin_2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist_coeffs)
            origin_2d = tuple(np.int32(origin_2d[0, 0]))

            cv2.circle(frame, origin_2d, 6, (0, 255, 255), -1)  # yellow point

            # --- 2) Project axis endpoints ---
            axis_len = tag_size * 1.5

            axes_3d = np.float32([
                [axis_len, 0, 0],   # X → red
                [0, axis_len, 0],   # Y → green
                [0, 0, axis_len]    # Z → blue
            ])

            axes_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist_coeffs)
            axes_2d = axes_2d.reshape(-1, 2).astype(int)

            cv2.line(frame, origin_2d, tuple(axes_2d[0]), (0, 0, 255), 3)   # X = red
            cv2.line(frame, origin_2d, tuple(axes_2d[1]), (0, 255, 0), 3)   # Y = green
            cv2.line(frame, origin_2d, tuple(axes_2d[2]), (255, 0, 0), 3)   # Z = blue


        out.write(frame)
    
    
def augment_tag_planes(rel_pose_map):
    internal_tag_id_rel_pose_map = []
    # Make the group1 quat to be (0.9239, 0.3827, 0, 0) and keep the position same.
    for tag_id, pos, quat in rel_pose_map:
        if (tag_id in group1):
            new_quat = 0.09239, 0.3827, 0.0, 0.0
            internal_tag_id_rel_pose_map.append((tag_id, pos, new_quat))
        elif (tag_id in group2):
            new_quat = 0.7071, 0.0, 0.7071, 0.0
            internal_tag_id_rel_pose_map.append((tag_id, pos, new_quat))
        else:
            print("Tag id not in any group:", tag_id) #should only print once for the static_tag_id.

    return internal_tag_id_rel_pose_map

def fit_plane(points):
        centroid = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - centroid)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        return centroid, normal

def plot_plane(ax, centroid, normal, color, label):
    xx, yy = np.meshgrid(
        np.linspace(centroid[0]-0.1, centroid[0]+0.1, 10),
        np.linspace(centroid[1]-0.1, centroid[1]+0.1, 10)
    )
    d = -centroid.dot(normal)
    zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color=color)
    ax.text(*centroid, label, color=color, fontsize=10)

def point_to_plane_distance(pt, centroid, normal):
        return np.dot(pt - centroid, normal)

def project_to_plane(pt, centroid, normal):
    dist = point_to_plane_distance(pt, centroid, normal)
    return pt - dist * normal, dist

def plot_45_degree_planes(ax, static_normal, origin, angle_deg, color, label):
    angle_rad = np.radians(angle_deg)
    rotation_axis = np.array([1, 0, 0])
    rot_matrix = R.from_rotvec(angle_rad * rotation_axis).as_matrix()
    rotated_normal = rot_matrix @ static_normal
    xx, yy = np.meshgrid(
        np.linspace(origin[0] - 0.5, origin[0] + 0.5, 10),
        np.linspace(origin[1] - 0.5, origin[1] + 0.5, 10)
    )
    d = -origin.dot(rotated_normal)
    zz = (-rotated_normal[0] * xx - rotated_normal[1] * yy - d) / rotated_normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color=color)
    ax.text(*origin, label, color=color, fontsize=10)

def visualize_relative_poses(rel_pose_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    positions = {}
    for tag_id, pos, quat in rel_pose_map:
        positions[tag_id] = pos
        ax.scatter(pos[0], pos[1], pos[2], s=100, label=f'Tag {tag_id}')
        # Draw orientation axes
        axes = np.eye(3) * 0.02
        rot = R.from_quat(quat)
        rotated_axes = rot.apply(axes)
        for i in range(3):
            ax.plot([pos[0], pos[0] + rotated_axes[i, 0]],
                    [pos[1], pos[1] + rotated_axes[i, 1]],
                    [pos[2], pos[2] + rotated_axes[i, 2]],
                    color=['r', 'g', 'b'][i])

    # Static tag at origin
    ax.scatter(0, 0, 0, s=150, c='k', label='Static Tag (ID 4)')
    axes = np.eye(3) * 0.03
    for i in range(3):
        ax.plot([0, axes[i, 0]], [0, axes[i, 1]], [0, axes[i, 2]], color=['r', 'g', 'b'][i])

    # Plot a larger plane along the static pose at origin
    xx, zz = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-0.5, 0.5, 20))
    yy = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    ax.text(0, 0, 0, 'Static Plane', color='gray', fontsize=10)

    # # --- Add two planes at 45 degrees to the static plane using plot plane function
    # static_plane_normal = np.array([0, 1, 0])  # Normal vector of the static plane (Y=0)
    # plot_45_degree_planes(ax, static_plane_normal, np.array([0, 0, 0]), 45, 'yellow', '+45° Plane')
    # plot_45_degree_planes(ax, static_plane_normal, np.array([0, 0, 0]), -45, 'green', '-45° Plane')

    # Define groups
    group1 = [3,10,15,9,5,7,8,13,6]
    group2 = [2,14,12,11,1,0,17,16,18]

    # --- Fit both planes ---
    pts1 = np.array([positions[i] for i in group1 if i in positions])
    pts2 = np.array([positions[i] for i in group2 if i in positions])

    c1, n1 = fit_plane(pts1)
    c2, n2 = fit_plane(pts2)

    # Print the equations of the planes
    print(f"Plane Group 1: normal = {n1}, point = {c1}")
    print(f"Plane Group 2: normal = {n2}, point = {c2}")

    # Plot planes
    plot_plane(ax, c1, n1, 'cyan', 'Plane Group 1')
    plot_plane(ax, c2, n2, 'magenta', 'Plane Group 2')

    # --- Compute angle between planes ---
    angle_rad = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    print(f"Angle between planes: {angle_deg:.2f}°")


    # --- Compute angle between n1 and n2 ---
    angle_rad_1 = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle_deg_1 = np.degrees(angle_rad_1)
    print(f"Angle between Plane Group 1 and Plane Group 2: {angle_deg_1:.2f}°")
    
    # --- Compute angle between static plane (Y=0 plane) and n1 ---
    static_plane_normal = np.array([0, 1, 0])  # Normal vector of the static plane (Y=0)
    angle_rad_static_n1 = np.arccos(np.clip(np.dot(n1, static_plane_normal), -1.0, 1.0))
    angle_deg_static_n1 = np.degrees(angle_rad_static_n1)
    print(f"Angle between static plane and Plane Group 1: {angle_deg_static_n1:.2f}°")
    
    # --- Compute angle between static plane and n2 ---
    angle_rad_2 = np.arccos(np.clip(np.dot(n2, static_plane_normal), -1.0, 1.0))
    angle_deg_2 = np.degrees(angle_rad_2)
    print(f"Angle between static plane and Plane Group 2: {angle_deg_2:.2f}°")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # To plot x,y,z range to be same
    max_range = 0.6
    ax.set_xlim([-max_range/2, max_range/2])
    ax.set_ylim([-max_range/2, max_range/2])
    ax.set_zlim([-max_range/2, max_range/2])    
    # plt.axis('equal')
    plt.show()

    # --- Project points onto their respective planes ---
    def project_group(pts, centroid, normal, group_name):
        projected_pts, dists = [], []
        for p in pts:
            proj, dist = project_to_plane(p, centroid, normal)
            projected_pts.append(proj)
            dists.append(dist)
        projected_pts = np.array(projected_pts)
        dists = np.array(dists)
        # Orthonormal basis in plane
        u = np.cross(normal, np.array([1,0,0]))
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(normal, np.array([0,1,0]))
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        coords = np.stack([np.dot(projected_pts - centroid, u),
                           np.dot(projected_pts - centroid, v)], axis=1)
        # Plot 2D projection
        plt.figure()
        plt.title(f"{group_name} – projection and distances")
        plt.scatter(coords[:,0], coords[:,1], c=dists, cmap='coolwarm', s=80)
        for i, d in enumerate(dists):
            plt.text(coords[i,0], coords[i,1], f"{d:.3f}", fontsize=9, ha='center', va='bottom')
        plt.colorbar(label='Distance from plane (m)')
        plt.xlabel('u-axis (in-plane)')
        plt.ylabel('v-axis (in-plane)')
        
        plt.show()

    # Plot both groups
    project_group(pts1, c1, n1, "Group 1")
    project_group(pts2, c2, n2, "Group 2")

if __name__ == "__main__":
    csv_file = 'ref_data/tag_paths_ref.csv'
    rel_pose_map = calculate_relative_poses(csv_file)
    rel_pose_map = augment_tag_planes(rel_pose_map)

    # # Print the map
    # for tag_id, pos, quat in rel_pose_map:
    #     print(f"Tag ID: {tag_id}, Position: {pos}, Quaternion: {quat}")
    
    
    # visualize_relative_poses(rel_pose_map)

    # reProject_pose(rel_pose_map) : wrong video input rn.

