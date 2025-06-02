import h5py
import os
import numpy as np

CUT_SECONDS = {
    "indoor_flying1": (10, 70),
    "indoor_flying2": (10, 70),
    "indoor_flying3": (10, 70),
    "indoor_flying4": (10, 19.8),
    "outdoor_day1": (0, 60),
    "outdoor_day2": (100, 160),
}

def read_path_list(filename):
    files = []
    with open(filename, "r") as f:
        for line in f:
            files.append(line.strip())
    return files

# Out format is like HQF.
# f["images"][f"{frame_idx:06d}"] is a (H, W) array of uint8 with attrs["event_idx"]
# f["events/ts"], f["events/xs"], f["events/ys"], f["events/ps"] are (N) arrays

def convert_mvsec(in_path, in_flow_path, out_path):
    sequence_name = os.path.basename(in_path).split("_data")[0]
    begin_second, end_second = CUT_SECONDS[sequence_name]
    side = "left"
    
    with h5py.File(out_path, 'w') as of:
        with h5py.File(in_path, 'r') as f:
            timestamps = f["davis"][side]["image_raw_ts"][:]
            raw_frame_timestamps = f["davis"][side]["image_raw_ts"][:]
            base_time = timestamps[0]
            timestamps = timestamps - timestamps[0]

            start_frame = np.searchsorted(timestamps, begin_second)
            end_frame = np.searchsorted(timestamps, end_second)
            imgs = f["davis"][side]["image_raw"][start_frame:end_frame]
            N, H, W = imgs.shape
            
            ev_begin_idx = max(f["davis"][side]["image_raw_event_inds"][start_frame], 0)
            ev_end_idx = f["davis"][side]["image_raw_event_inds"][end_frame]
            events = f["davis"][side]["events"][ev_begin_idx:ev_end_idx]
            
            of.create_dataset("events/ts", data=f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 2], dtype=np.float64)
            of.create_dataset("events/xs", data=f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 0], dtype=np.uint16)
            of.create_dataset("events/ys", data=f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 1], dtype=np.uint16)

            ps = f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 3]
            ps = (ps + 1) / 2
            of.create_dataset("events/ps", data=(f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 3]+1)/2, dtype=np.uint8)

            all_event_ts = f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 2]

            for idx, img in enumerate(imgs):
                image_name = f"images/image{idx:09d}"
                of.create_dataset(image_name, data=img)

                of[image_name].attrs["event_idx"] = max(f["davis"][side]["image_raw_event_inds"][start_frame + idx] - ev_begin_idx, 0)
                of[image_name].attrs["timestamp"] = f["davis"][side]["image_raw_ts"][start_frame + idx]

            of.attrs["sensor_resolution"] = H, W
            of.attrs["num_events"] = events.shape[0]
            of.attrs["num_imgs"] = N
            of.attrs["data_source"] = "mvsec"
        
        with h5py.File(in_flow_path, 'r') as f:
            flow_timestamps = f["davis"]["left"]["depth_image_raw_ts"][:]
            flow_timestamps_norm = flow_timestamps - base_time
            # There are flow frames before any events.
            begin_flow_frame = np.searchsorted(flow_timestamps_norm, begin_second)
            end_flow_frame = np.searchsorted(flow_timestamps_norm, end_second)

            flow_to_ev_idx = np.searchsorted(all_event_ts, flow_timestamps)
            flow_to_img_idx = np.searchsorted(raw_frame_timestamps[start_frame:end_frame], flow_timestamps)
            print("flow_to_ev_idx:", flow_to_ev_idx[begin_flow_frame:end_flow_frame])
            print("flow_to_img_idx:", flow_to_img_idx[begin_flow_frame:end_flow_frame])

            for i in range(begin_flow_frame, end_flow_frame):
                flow_name = f"flow/flow{i-begin_flow_frame:09d}"

                depth = f["davis"]["left"]["depth_image_raw"][i]
                flow = f["davis"]["left"]["flow_dist"][i]
                # Mark pixels where depth is NaN as NaN in flow
                flow = np.where(np.isnan(depth), np.nan, flow)
                of.create_dataset(flow_name, data=flow, dtype=np.float32)

                of[flow_name].attrs["event_idx"] = flow_to_ev_idx[i]
                of[flow_name].attrs["timestamp"] = flow_timestamps[i]
                of[flow_name].attrs["image_idx"] = flow_to_img_idx[i]


def convert_mvsec_noflow(in_path, out_path):
    sequence_name = os.path.basename(in_path).split("_data")[0]
    begin_second, end_second = CUT_SECONDS[sequence_name]
    side = "left"
    
    with h5py.File(out_path, 'w') as of:
        with h5py.File(in_path, 'r') as f:
            timestamps = f["davis"][side]["image_raw_ts"][:]
            base_time = timestamps[0]
            timestamps = timestamps - timestamps[0]

            start_frame = np.searchsorted(timestamps, begin_second)
            end_frame = np.searchsorted(timestamps, end_second)
            imgs = f["davis"][side]["image_raw"][start_frame:end_frame]
            N, H, W = imgs.shape
            
            ev_begin_idx = max(f["davis"][side]["image_raw_event_inds"][start_frame], 0)
            ev_end_idx = f["davis"][side]["image_raw_event_inds"][end_frame]
            events = f["davis"][side]["events"][ev_begin_idx:ev_end_idx]
            
            of.create_dataset("events/ts", data=f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 2], dtype=np.float64)
            of.create_dataset("events/xs", data=f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 0], dtype=np.uint16)
            of.create_dataset("events/ys", data=f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 1], dtype=np.uint16)

            ps = f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 3]
            ps = (ps + 1) / 2
            of.create_dataset("events/ps", data=(f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 3]+1)/2, dtype=np.uint8)

            all_event_ts = f["davis"][side]["events"][ev_begin_idx:ev_end_idx, 2]

            for idx, img in enumerate(imgs):
                image_name = f"images/image{idx:09d}"
                of.create_dataset(image_name, data=img)

                of[image_name].attrs["event_idx"] = max(f["davis"][side]["image_raw_event_inds"][start_frame + idx] - ev_begin_idx, 0)
                of[image_name].attrs["timestamp"] = f["davis"][side]["image_raw_ts"][start_frame + idx]

            of.attrs["sensor_resolution"] = H, W
            of.attrs["num_events"] = events.shape[0]
            of.attrs["num_imgs"] = N
            of.attrs["data_source"] = "mvsec"


in_base_path = "/mnt/ssd/mvsec/"
in_path_list = [
    ("indoor_flying/indoor_flying1_data.hdf5", "indoor_flying/indoor_flying1_gt.hdf5"),
    ("indoor_flying/indoor_flying2_data.hdf5", "indoor_flying/indoor_flying2_gt.hdf5"),
    ("indoor_flying/indoor_flying3_data.hdf5", "indoor_flying/indoor_flying3_gt.hdf5"),
    ("outdoor_day/outdoor_day1_data.hdf5", "outdoor_day/outdoor_day1_gt.hdf5"),
    ("outdoor_day/outdoor_day2_data.hdf5", "outdoor_day/outdoor_day2_gt.hdf5"),
]

# Convert with optical flow
out_base_path = "/mnt/ssd/MVSEC_wflow/"
os.makedirs(out_base_path, exist_ok=True)
out_path_list = [
    "indoor_flying1.h5",
    "indoor_flying2.h5",
    "indoor_flying3.h5",
    "outdoor_day1.h5",
    "outdoor_day2.h5",
]

for in_paths, out_path in zip(in_path_list[:], out_path_list[:]):
    in_path = os.path.join(in_base_path, in_paths[0])
    in_flow_path = os.path.join(in_base_path, in_paths[1])
    out_path = os.path.join(out_base_path, out_path)
    print(in_path, in_flow_path, out_path)
    convert_mvsec(in_path, in_flow_path, out_path)


# Convert without optical flow
out_base_path_noflow = "/mnt/ssd/MVSEC_cut/"
os.makedirs(out_base_path_noflow, exist_ok=True)
in_path_list_noflow = [
    ("indoor_flying/indoor_flying1_data.hdf5",),
    ("indoor_flying/indoor_flying2_data.hdf5",),
    ("indoor_flying/indoor_flying3_data.hdf5",),
    ("indoor_flying/indoor_flying4_data.hdf5",),  # This one has no flow
    ("outdoor_day/outdoor_day1_data.hdf5",),
    ("outdoor_day/outdoor_day2_data.hdf5",),
]
out_path_list_noflow = [
    "indoor_flying1.h5",
    "indoor_flying2.h5",
    "indoor_flying3.h5",
    "indoor_flying4.h5",  # This one has no flow
    "outdoor_day1.h5",
    "outdoor_day2.h5",
]

for in_paths, out_path in zip(in_path_list_noflow, out_path_list_noflow):
    in_path = os.path.join(in_base_path, in_paths[0])
    out_path = os.path.join(out_base_path_noflow, out_path)
    print(in_path, out_path)
    convert_mvsec_noflow(in_path, out_path)