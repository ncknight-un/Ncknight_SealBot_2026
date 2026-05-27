# Read the frames data from jpg files and timestamps from csv file, to form a mp4 video. 

import cv2
import csv
import os
import glob
from natsort import natsorted

# ---------- CONFIG ----------
frames_dir = "frames_output"
csv_filename = os.path.join(frames_dir, "timestamps.csv")
output_video = "output_video.mp4"
frame_rate = 30  # Desired frame rate for the output video

# ---------- READ TIMESTAMPS ----------
timestamps = []
with open(csv_filename, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        timestamps.append(float(row["timestamp_sec"]))

# ---------- READ FRAMES ----------
frame_files = natsorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
frames = [cv2.imread(f) for f in frame_files]

# ---------- WRITE VIDEO ----------
if not frames:
    print("No frames found to create video.")
    exit(1)
height, width, layers = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
for frame in frames:
    video.write(frame)
video.release()
print(f"Video saved to {output_video}")

