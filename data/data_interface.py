from utils.util import get_obj_from_str
from torch.utils.data import ConcatDataset
import pandas as pd
from tqdm import tqdm

def make_concat_dataset(configs):
    data_file = configs["data_file"]
    class_name = configs["class_name"]
    dataset_type = get_obj_from_str(class_name)
    data_paths = pd.read_csv(data_file, header=None).values.flatten().tolist()

    begin_seq = configs.get("begin_seq", 0)
    end_seq = configs.get("end_seq", len(data_paths))
    data_paths = data_paths[begin_seq:end_seq]
    
    dataset_list = []
    print('Concatenating {} datasets'.format(dataset_type))
    for data_path in tqdm(data_paths):
        dataset_list.append(dataset_type(data_path, configs))
    print("Total samples: ", sum([len(d) for d in dataset_list]))
    return ConcatDataset(dataset_list)

def make_concat_multi_dataset(configs):
    datasets = []
    for config in configs:
        datasets.append(make_concat_dataset(config))
    return ConcatDataset(datasets)