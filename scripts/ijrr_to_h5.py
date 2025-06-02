import os
import sys
import h5py
import numpy as np
import glob
import cv2
import tqdm

def unzip(file_path):
    # Use the system unzip command to unzip it into the current directory
    # If the zip contains a folter named "name", unzip it directly. Else, unzip it into file_dir/name.
    file_dir = os.path.dirname(file_path)
    name = os.path.basename(file_path).split(".")[0]
    out_dir = os.path.join(file_dir, name)
    os.system(f"unzip -d {out_dir} {file_path}")
    # If this creates a directory file_dir/name/name, move the contents of file_dir/name/name to file_dir/name
    if os.path.isdir(os.path.join(out_dir, name)):
        os.system(f"mv {os.path.join(out_dir, name)}/* {out_dir}")
        os.system(f"rm -r {os.path.join(out_dir, name)}")

CUT_SECONDS = {
    "boxes_6dof": (5, 20),
    "calibration": (5, 20),
    "dynamic_6dof": (5, 20),
    "office_zigzag": (5, 12),
    "poster_6dof": (5, 20),
    "shapes_6dof": (5, 20),
    "slider_depth": (1, 2.5)
}

IN_DIR = "/mnt/ssd/IJRR"
OUT_DIR = "/mnt/ssd/IJRR_cut"
os.makedirs(OUT_DIR, exist_ok=True)

for seq_name in CUT_SECONDS.keys():
    zip_path = f"{IN_DIR}/{seq_name}.zip"
    unzip(zip_path)

    out_h5path = f"{OUT_DIR}/{seq_name}.h5"
    in_root = f"{IN_DIR}/{seq_name}"

    img_timestamp_txt = f"{in_root}/images.txt"
    # In the txt is N rows. Each row is timestamp + image filepath, such as :
    # 1468941032.255472635 images/frame_00000000.png
    timestamps = []
    img_paths = []
    with open(img_timestamp_txt, "r") as f:
        for line in f:
            timestamp, img_path = line.strip().split(" ")
            timestamps.append(float(timestamp))
            img_paths.append(img_path)
    # The events are stored in a txt file. Each line is [t x y p], such as:
    # 1468941032.229165635 128 154 1
    events_txt = f"{in_root}/events.txt"
    events = np.loadtxt(events_txt, dtype=np.float64)

    ts = events[:, 0]

    event_begin_idx = np.searchsorted(ts, CUT_SECONDS[seq_name][0] + timestamps[0])
    event_end_idx = np.searchsorted(ts, CUT_SECONDS[seq_name][1] + timestamps[0])
    print("timestamps[0]", timestamps[0])
    print("CUT_SECONDS[seq_name][1]", CUT_SECONDS[seq_name][1])
    print("ts[0]", ts[0])
    print("event_begin_idx", event_begin_idx, "event_end_idx", event_end_idx)
    image_begin_idx = np.searchsorted(timestamps, CUT_SECONDS[seq_name][0] + timestamps[0])
    image_end_idx = np.searchsorted(timestamps, CUT_SECONDS[seq_name][1] + timestamps[0])

    img_ev_idx = []
    for i in range(image_begin_idx, image_end_idx):
        img_ev_idx.append(np.searchsorted(ts[event_begin_idx:event_end_idx], timestamps[i]))
    
    # Extract the event data and images first
    event_xs = events[event_begin_idx:event_end_idx, 1].astype(np.uint16)
    event_ys = events[event_begin_idx:event_end_idx, 2].astype(np.uint16)
    event_ts = events[event_begin_idx:event_end_idx, 0].astype(np.float64)
    event_ps = events[event_begin_idx:event_end_idx, 3].astype(np.uint8)
    
    images = []
    for img_path in img_paths[image_begin_idx:image_end_idx]:
        images.append(cv2.imread(f"{in_root}/{img_path}", cv2.IMREAD_GRAYSCALE))
    images = np.stack(images)
    N, H, W = images.shape
    
    # Output in HQF format directly
    with h5py.File(out_h5path, "w") as f:
        # Save metadata as attributes
        f.attrs["sensor_resolution"] = (H, W)
        f.attrs["num_events"] = event_ts.shape[0]
        f.attrs["num_imgs"] = N
        f.attrs["data_source"] = "ijrr"
        
        # Save event data
        f.create_dataset("events/xs", data=event_xs)
        f.create_dataset("events/ys", data=event_ys)
        f.create_dataset("events/ts", data=event_ts)
        f.create_dataset("events/ps", data=event_ps)
        
        # Save images with proper attributes
        img_timestamps = np.array(timestamps[image_begin_idx:image_end_idx])
        for idx in range(N):
            image_name = f"images/image{idx:09d}"
            f.create_dataset(image_name, data=images[idx])
            f[image_name].attrs["event_idx"] = img_ev_idx[idx]
            f[image_name].attrs["timestamp"] = img_timestamps[idx]
        
    print(f"Processed {seq_name} - saved to {out_h5path}")



