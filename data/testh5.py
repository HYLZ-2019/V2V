import torch
import h5py
import os
import numpy as np
import cv2
from event_voxel_builder import EventVoxelBuilder

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import data_sources

# The authors of HQF have already cut the time ranges.

class TestH5Dataset(torch.utils.data.Dataset):
    '''
    Test with frames from HQF-like format.
    '''
    def __init__(self, h5_path, configs):
        self.h5_path = h5_path
        # The h5_path should be like HQF_h5/bike_bay_hdr.h5. The corresponding sequence_name is hqf_bike_bay_hdr.
        self.sequence_name = os.path.basename(h5_path).split(".")[0]

        self.configs = configs
        self.dataset_name = configs.get("dataset_name", "hqf")
        self.sequence_length = configs.get("sequence_length", 40)
        self.warm_up_length = configs.get("warm_up_length", 0)
        self.max_samples = configs.get("max_samples", None)

        self.num_bins = configs.get("num_bins", 5)
        self.interpolate_bins = configs.get("interpolate_bins", False)
        self.image_range = configs.get("image_range", 255)
        assert self.image_range in [255, 1], "image_range should be 255 or 1."

        with h5py.File(self.h5_path, 'r') as f:
            self.img_keys = sorted(f["images"].keys())
            self.total_frame_cnt = len(self.img_keys)
            img_shape = f["images"][self.img_keys[0]].shape
            self.H = img_shape[0]#180
            self.W = img_shape[1]#240
            
            self.samples = []
            # Sample = (start_idx, real_start_idx, end_idx)
            # regin: the index of the first input frame (might be for warm up)
            # real_start_idx: The first non-warmup frame is begin+real_start_idx
            # end_idx: the index of the last input frame
            for i in range(0, self.total_frame_cnt-1, self.sequence_length-self.warm_up_length):
                begin = max(0, i - self.warm_up_length)
                end_idx = min(self.total_frame_cnt-1, begin + self.sequence_length)
                self.samples.append((begin, i-begin, end_idx))

        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]
        
        self.output_additional_frame = configs.get("output_additional_frame", False)
        self.output_additional_evs = configs.get("output_additional_evs", False)
    
    def __len__(self):
        return len(self.samples)
    
    def make_voxel(self, evs):
        voxel = np.zeros((self.num_bins, self.H, self.W))
        ts, xs, ys, ps = evs
        if ts.shape[0] == 0:
            return voxel
        
        # ps of hqf h5 file are in {0, 1}.
        ps = ps.astype(np.int8) * 2 - 1
        ts = ((ts - ts[0]) * 1e6).astype(np.int64)
            
        if not self.interpolate_bins:
            t_per_bin = (ts[-1] + 0.001) / self.num_bins
            bin_idx = np.floor(ts / t_per_bin).astype(np.uint8)
            np.add.at(voxel, (bin_idx, ys, xs), ps)
        else:
            # Interpolate the events to the bins.
            dt = ts[-1] - ts[0]
            t_norm = (ts - ts[0]) / (dt + 0.0001) * (self.num_bins - 1)
            for bi in range(self.num_bins):
                weights = np.maximum(0, 1.0 - np.abs(t_norm - bi))
                np.add.at(voxel, (bi, ys, xs), weights*ps)

        # Different to hot pixel detection in e2vid++.
        # e2vid++ version: Reconstructs image then selects K max pixels. Then it zeros all events on the pixel. Obviously it isn't a good algorithm so e2vid++ doesn't use it in testing.
        # if self.filter_hot_events:
        #     hot_thres = max(20, np.abs(pos_max) * 5)
        #     total_ev_cnt = np.sum(np.abs(voxel), axis=0)
        #     not_hot_mask = (total_ev_cnt < hot_thres)[np.newaxis, ...]
        #     voxel = voxel * not_hot_mask
        
        return voxel
        
    def get_img(self, f, idx):
        img = f["images"][self.img_keys[idx]][()]
        return img
    
    def __getitem__(self, idx):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        begin, real_begin, end = self.samples[idx]
        sequence = []

        with h5py.File(self.h5_path, 'r') as f:
            for img_idx in range(begin, end):
                item = {}
                img = self.get_img(f, img_idx+1)
                item["frame"] = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

                ev_begin_idx = f["images"][self.img_keys[img_idx]].attrs["event_idx"]
                ev_end_idx = f["images"][self.img_keys[img_idx+1]].attrs["event_idx"]
                ts = f["events/ts"][ev_begin_idx:ev_end_idx]
                xs = f["events/xs"][ev_begin_idx:ev_end_idx]
                ys = f["events/ys"][ev_begin_idx:ev_end_idx]
                ps = f["events/ps"][ev_begin_idx:ev_end_idx]
                voxel = self.make_voxel([ts, xs, ys, ps])
                item["events"] = torch.tensor(voxel, dtype=torch.float32)

                item["flow"] = torch.zeros((2, self.H, self.W))
                item["data_source_idx"] = data_sources.index(self.dataset_name.lower())
                item["sequence_name"] = self.sequence_name
                item["real_begin_idx"] = real_begin
                item["frame_idx"] = img_idx

                sequence.append(item)

            # Additional frame for pseudo ground truth optical flow.
            if self.output_additional_frame:
                first_frame = self.get_img(f, begin)
                first_frame = torch.tensor(first_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # Additional events in the front, for ERAFT inference
            if self.output_additional_evs:
                pre_idx = max(0, begin-1)
                pre_ev_begin_idx = f["images"][self.img_keys[pre_idx]].attrs["event_idx"]
                ev_begin_idx = f["images"][self.img_keys[begin]].attrs["event_idx"]
                first_voxel = self.make_voxel([
                    f["events/ts"][pre_ev_begin_idx:ev_begin_idx],
                    f["events/xs"][pre_ev_begin_idx:ev_begin_idx],
                    f["events/ys"][pre_ev_begin_idx:ev_begin_idx],
                    f["events/ps"][pre_ev_begin_idx:ev_begin_idx]
                ])
                first_voxel = torch.tensor(first_voxel, dtype=torch.float32).unsqueeze(0)


        all_frames = torch.stack([item["frame"] for item in sequence], dim=0)
        if self.output_additional_frame:
            all_frames = torch.cat([first_frame, all_frames], dim=0)

        # The standard real-data-h5 should have [0, 255] range.
        if self.image_range == 1:
            all_frames = all_frames / 255.0

        all_events = torch.stack([item["events"] for item in sequence], dim=0)
        if self.output_additional_evs:
            all_events = torch.cat([first_voxel, all_events], dim=0)

        all_data_source_idx = torch.tensor(sequence[0]["data_source_idx"], dtype=torch.int64)
        all_sequence_name = [item["sequence_name"] for item in sequence]
        all_real_begin_idx = torch.tensor([item["real_begin_idx"] for item in sequence], dtype=torch.int64)
        all_frame_idx = torch.tensor([item["frame_idx"] for item in sequence], dtype=torch.int64)
        
        new_sequence = {
            "frame": all_frames,
            "events": all_events,
            "data_source_idx": all_data_source_idx,
            "sequence_name": all_sequence_name,
            "real_begin_idx": all_real_begin_idx,
            "frame_idx": all_frame_idx
        }
        return new_sequence

class TestH5FlowDataset(TestH5Dataset):
    '''
    Test with frames from MVSEC-like format.
    '''
    def __init__(self, h5_path, configs):
        self.h5_path = h5_path
        # The h5_path should be like HQF_h5/bike_bay_hdr.h5. The corresponding sequence_name is hqf_bike_bay_hdr.
        self.sequence_name = os.path.basename(h5_path).split(".")[0]

        self.configs = configs
        self.dataset_name = configs.get("dataset_name", "mvsec")
        self.sequence_length = configs.get("sequence_length", 40)
        self.max_samples = configs.get("max_samples", None)

        self.num_bins = configs.get("num_bins", 5)
        self.interpolate_bins = configs.get("interpolate_bins", False)
        self.image_range = configs.get("image_range", 255)
        assert self.image_range in [255, 1], "image_range should be 255 or 1."

        with h5py.File(self.h5_path, 'r') as f:
            self.img_keys = sorted(f["images"].keys())
            self.flow_keys = sorted(f["flow"].keys())
            self.total_frame_cnt = len(self.flow_keys)
            img_shape = f["images"][self.img_keys[0]].shape
            self.H = img_shape[0]
            self.W = img_shape[1]
            
            self.samples = []
            # Sample = (start_idx, end_idx)
            for i in range(0, self.total_frame_cnt-1, self.sequence_length):
                begin = i
                end_idx = min(self.total_frame_cnt-1, begin + self.sequence_length)
                self.samples.append((begin, end_idx))

        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]
        
        self.output_additional_frame = configs.get("output_additional_frame", False)
        self.output_additional_evs = configs.get("output_additional_evs", False)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        begin, end = self.samples[idx]
        sequence = []

        with h5py.File(self.h5_path, 'r') as f:
            for flow_idx in range(begin, end):
                item = {}

                flow_item = f["flow"][self.flow_keys[flow_idx+1]]
                prev_flow_item = f["flow"][self.flow_keys[flow_idx]]

                img_idx = flow_item.attrs["image_idx"]
                img_idx = min(img_idx, len(self.img_keys)-1)
                img = self.get_img(f, img_idx)
                item["frame"] = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

                ev_begin_idx = prev_flow_item.attrs["event_idx"]
                ev_end_idx = flow_item.attrs["event_idx"]

                ts = f["events/ts"][ev_begin_idx:ev_end_idx]
                xs = f["events/xs"][ev_begin_idx:ev_end_idx]
                ys = f["events/ys"][ev_begin_idx:ev_end_idx]
                ps = f["events/ps"][ev_begin_idx:ev_end_idx]
                voxel = self.make_voxel([ts, xs, ys, ps])
                item["events"] = torch.tensor(voxel, dtype=torch.float32)

                item["flow"] = torch.tensor(flow_item[()])

                item["data_source_idx"] = data_sources.index(self.dataset_name.lower())
                item["sequence_name"] = self.sequence_name
                item["frame_idx"] = img_idx

                sequence.append(item)
            
            if self.output_additional_frame:
                prev_flow_item = f["flow"][self.flow_keys[begin]]
                first_img_idx = prev_flow_item.attrs["image_idx"]
                first_frame = self.get_img(f, first_img_idx)

            if self.output_additional_evs:
                pre_flow_idx = max(0, begin-1)
                pre_event_idx = f["flow"][self.flow_keys[pre_flow_idx]].attrs["event_idx"]
                end_event_idx = f["flow"][self.flow_keys[begin]].attrs["event_idx"]
                first_voxel = self.make_voxel([
                    f["events/ts"][pre_event_idx:end_event_idx],
                    f["events/xs"][pre_event_idx:end_event_idx],
                    f["events/ys"][pre_event_idx:end_event_idx],
                    f["events/ps"][pre_event_idx:end_event_idx]
                ])
                first_voxel = torch.tensor(first_voxel, dtype=torch.float32).unsqueeze(0)

        all_frames = torch.stack([item["frame"] for item in sequence], dim=0)

        if self.output_additional_frame:
            first_frame = torch.tensor(first_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            all_frames = torch.cat([first_frame, all_frames], dim=0)

        # The standard real-data-h5 should have [0, 255] range.
        if self.image_range == 1:
            all_frames = all_frames / 255.0

        all_events = torch.stack([item["events"] for item in sequence], dim=0)
        if self.output_additional_evs:
            all_events = torch.cat([first_voxel, all_events], dim=0)

        all_flow = torch.stack([item["flow"] for item in sequence], dim=0)
        all_data_source_idx = torch.tensor(sequence[0]["data_source_idx"], dtype=torch.int64)
        all_sequence_name = [item["sequence_name"] for item in sequence]
        all_frame_idx = torch.tensor([item["frame_idx"] for item in sequence], dtype=torch.int64)
        
        new_sequence = {
            "frame": all_frames,
            "events": all_events,
            "flow": all_flow,
            "data_source_idx": all_data_source_idx,
            "sequence_name": all_sequence_name,
            "frame_idx": all_frame_idx
        }
        return new_sequence

class TestH5EventDataset(TestH5Dataset):
    '''
    Difference from TestH5Dataset: Returns N*4 events instead of stacked voxels.
    '''
    def __getitem__(self, idx):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        begin, real_begin, end = self.samples[idx]
        sequence = []

        with h5py.File(self.h5_path, 'r') as f:
            for img_idx in range(begin, end):
                item = {}
                img = self.get_img(f, img_idx+1)
                item["frame"] = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

                ev_begin_idx = f["images"][self.img_keys[img_idx]].attrs["event_idx"]
                ev_end_idx = f["images"][self.img_keys[img_idx+1]].attrs["event_idx"]

                # Important: in some h5 files, using np.float32 will lose all the precision.
                ts = f["events/ts"][ev_begin_idx:ev_end_idx].astype(np.float64)
                xs = f["events/xs"][ev_begin_idx:ev_end_idx].astype(np.float64)
                ys = f["events/ys"][ev_begin_idx:ev_end_idx].astype(np.float64)
                ps = f["events/ps"][ev_begin_idx:ev_end_idx].astype(np.float64)
                ps = ps * 2 - 1 # [-1, 1]
      
                # events: shape: torch.Size([n, 5]), [x, y, t, p, b]
                # asserting batch size is 1
                evs = np.stack([xs, ys, ts, ps, np.zeros_like(ps, dtype=np.float64)], axis=1).copy()
                item["events"] = torch.tensor(evs, dtype=torch.float64)
                #print("in item:", item["events"][0, 2], item["events"][-1, 2])
                
                if item["events"].shape[0] == 0:
                    item["events"] = torch.zeros((1, 5), dtype=torch.float64)

                item["flow"] = torch.zeros((2, self.H, self.W))
                item["data_source_idx"] = data_sources.index(self.dataset_name.lower())
                item["sequence_name"] = self.sequence_name
                item["real_begin_idx"] = real_begin
                item["frame_idx"] = img_idx

                sequence.append(item)

            # Additional frame for pseudo ground truth optical flow.
            if self.output_additional_frame:
                first_frame = self.get_img(f, begin)
                first_frame = torch.tensor(first_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        all_frames = torch.stack([item["frame"] for item in sequence], dim=0)
        if self.output_additional_frame:
            all_frames = torch.cat([first_frame, all_frames], dim=0)

        # The standard real-data-h5 should have [0, 255] range.
        if self.image_range == 1:
            all_frames = all_frames / 255.0

        all_events = [item["events"] for item in sequence]
        all_data_source_idx = torch.tensor(sequence[0]["data_source_idx"], dtype=torch.int64)
        all_sequence_name = [item["sequence_name"] for item in sequence]
        all_real_begin_idx = torch.tensor([item["real_begin_idx"] for item in sequence], dtype=torch.int64)
        all_frame_idx = torch.tensor([item["frame_idx"] for item in sequence], dtype=torch.int64)
        
        new_sequence = {
            "frame": all_frames,
            "events": all_events,
            "data_source_idx": all_data_source_idx,
            "sequence_name": all_sequence_name,
            "real_begin_idx": all_real_begin_idx,
            "frame_idx": all_frame_idx
        }
        return new_sequence


class TestH5CacheDataset(torch.utils.data.Dataset):
    '''
    Pre-built voxel caches from TestH5Dataset.
    Datasets can be converted from TestH5Dataset to TestH5CacheDataset using scripts/testh5_to_voxel_cache.py.
    '''
    def __init__(self, h5_path, configs):
        self.h5_path = h5_path
        # The h5_path should be like HQF_h5/bike_bay_hdr.h5. The corresponding sequence_name is hqf_bike_bay_hdr.
        self.sequence_name = os.path.basename(h5_path).split(".")[0]

        self.configs = configs
        self.dataset_name = configs.get("dataset_name", "hqf")
        self.sequence_length = configs.get("sequence_length", 40)

        self.num_bins = configs.get("num_bins", 5)
        self.interpolate_bins = configs.get("interpolate_bins", False)

        with h5py.File(self.h5_path, 'r') as f:
            assert self.num_bins == f.attrs["num_bins"]
            assert self.interpolate_bins == f.attrs["interpolate_bins"]

            self.total_frame_cnt = f["frames"].shape[0]
            img_shape = f["frames"].shape[1:]
            self.H = img_shape[0]#180
            self.W = img_shape[1]#240
            
            self.samples = []
            # Sample = (start_idx, end_idx)
            for i in range(0, self.total_frame_cnt, self.sequence_length):
                begin = i
                end_idx = min(self.total_frame_cnt, begin + self.sequence_length)
                self.samples.append((begin, end_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        begin, end = self.samples[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            all_frames = f["frames"][begin:end]
            all_events = f["events"][begin:end]

        all_frames = torch.tensor(all_frames)
        all_events = torch.tensor(all_events)

        data_source_idx = data_sources.index(self.dataset_name.lower())
        all_data_source_idx = torch.tensor([data_source_idx for i in range(begin, end)], dtype=torch.int64)
        all_sequence_name = [self.sequence_name for i in range(begin, end)]
       
        new_sequence = {
            "frame": all_frames,
            "events": all_events,
            "data_source_idx": all_data_source_idx,
            "sequence_name": all_sequence_name,
        }
        return new_sequence

class FPS_H5Dataset(TestH5Dataset):
    '''
    Test with frames from HQF-like format.
    '''
    def __init__(self, h5_path, configs):
        self.h5_path = h5_path
        # The h5_path should be like HQF_h5/bike_bay_hdr.h5. The corresponding sequence_name is hqf_bike_bay_hdr.
        self.sequence_name = os.path.basename(h5_path).split(".")[0]

        self.configs = configs
        self.dataset_name = configs.get("dataset_name", "hqf")
        self.sequence_length = configs.get("sequence_length", 40)
        self.warm_up_length = configs.get("warm_up_length", 0)

        self.num_bins = configs.get("num_bins", 5)
        self.interpolate_bins = configs.get("interpolate_bins", False)
        self.FPS = configs.get("FPS", 100)
        self.H = configs.get("H", 260)
        self.W = configs.get("W", 346)

        with h5py.File(self.h5_path, 'r') as f:
            min_t = f["events/ts"][0]
            max_t = f["events/ts"][-1]
            self.total_frame_cnt = int((max_t - min_t) * self.FPS)

            border_timestamps = np.linspace(min_t, max_t, self.total_frame_cnt+1)
            self.event_idx = np.searchsorted(f["events/ts"], border_timestamps)
            
            self.samples = []
            # Sample = (start_idx, end_idx)
            for i in range(0, self.total_frame_cnt-1, self.sequence_length):
                begin_idx = i
                end_idx = min(self.total_frame_cnt-1, begin_idx + self.sequence_length)
                self.samples.append((begin_idx, end_idx))
    
    def __getitem__(self, idx):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        begin, end = self.samples[idx]
        sequence = []

        with h5py.File(self.h5_path, 'r') as f:
            for img_idx in range(begin, end):
                item = {}                
                ev_begin_idx = self.event_idx[img_idx]
                ev_end_idx = self.event_idx[img_idx+1]
                ts = f["events/ts"][ev_begin_idx:ev_end_idx]
                xs = f["events/xs"][ev_begin_idx:ev_end_idx]
                ys = f["events/ys"][ev_begin_idx:ev_end_idx]
                ps = f["events/ps"][ev_begin_idx:ev_end_idx]
                voxel = self.make_voxel([ts, xs, ys, ps])
                item["events"] = torch.tensor(voxel, dtype=torch.float32)
                item["data_source_idx"] = data_sources.index(self.dataset_name.lower())
                item["sequence_name"] = self.sequence_name

                sequence.append(item)

        all_events = torch.stack([item["events"] for item in sequence], dim=0)
        all_data_source_idx = torch.tensor(sequence[0]["data_source_idx"], dtype=torch.int64)
        all_sequence_name = [item["sequence_name"] for item in sequence]

        new_sequence = {
            "events": all_events,
            "data_source_idx": all_data_source_idx,
            "sequence_name": all_sequence_name,
        }
        return new_sequence