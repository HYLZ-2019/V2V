# V2V: Scaling Event-Based Vision through Efficient Video-to-Voxel Simulation

This is the code for the paper ["V2V: Scaling Event-Based Vision through Efficient Video-to-Voxel Simulation"](https://arxiv.org/abs/2505.16797). We provide code for training and testing several models for event-based video reconstruction and optical flow estimation.

## Experiment Configurations

To train or test with the code, you need an **experiment configuration**, such as `config/train_v2v_e2vid_10k.yaml`. In each configuration file, the core parameter is the `experiment_name`. 

To train with configuration `{config_file}`, execute `python train.py {config_file}`. The training logs will be saved in `tensorboard_logs/{experiment_name}`, and the checkpoints will be saved in `checkpoints/{experiment_name}`. The paths to the saved checkpoints will be appended to `ckpt_paths/{experiment_name}.txt`.

To test with the configuration, execute `python test_e2vid.py {config_file}` for video reconstruction, or `python test_flow.py {config_file}` for optical flow estimation. The test will read the checkpoint path from the last line of `ckpt_paths/{experiment_name}.txt`, and load the corresponding checkpoint. You can specify the output directory with the `test_output_dir` parameter in the configuration file.

Since different checkpoints may have different performances, after training, you can use `python scripts/select_best_checkpoint.py {experiment_name} {check_val_every_n_epoch}`, which will output the checkpoint path with least validation loss. Then, manually add it to the last line of `ckpt_paths/{experiment_name}.txt` for testing.

We provide the following configurations:

| Config                             | Task   | Model       | Train code | Test checkpoint |
|------------------------------------|--------|-------------|------------|-----------------|
| test_e2vid++_original.yaml         | E2VID  | E2VID       | ×          | √               |
| test_etnet_original.yaml           | E2VID  | ETNet       | ×          | √               |
| test_hypere2vid_original.yaml      | E2VID  | HyperE2VID  | ×          | √               |
| test_nernet_original.yaml          | E2VID  | NerNet      | ×          | √               |
| test_evflow_original.yaml          | Flow   | EvFlow      | ×          | √               |
| test_eraft_original.yaml           | Flow   | ERAFT       | ×          | √               |
| train_v2v_e2vid_10k.yaml           | E2VID  | E2VID       | √          | √               |
| train_v2v_etnet_10k.yaml           | E2VID  | ETNet       | √          | √               |
| train_v2v_hyper_10k.yaml           | E2VID  | HyperE2VID  | √          | √               |
| train_v2v_evflow_10k.yaml          | Flow   | EvFlow      | √          | √               |
| train_v2v_eraft_10k.yaml           | Flow   | ERAFT       | √          | √               |
| train_ablation_e2vid_esim.yaml     | E2VID  | E2VID       | √          | ×               |
| train_ablation_e2vid_10k_fixed.yaml| E2VID  | E2VID       | √          | ×               |
| train_ablation_e2vid_filtered.yaml | E2VID  | E2VID       | √          | ×               |
| train_ablation_e2vid_hdr.yaml      | E2VID  | E2VID       | √          | ×               |

The code for the model designs [E2VID](https://github.com/TimoStoff/event_cnn_minimal), [ETNet](https://github.com/WarranWeng/ET-Net), [HyperE2VID](https://github.com/ercanburak/HyperE2VID/), [NerNet](https://github.com/Liu-haoyue/NER-Net), [EvFlow](https://github.com/TimoStoff/event_cnn_minimal) and [ERAFT](https://github.com/uzh-rpg/E-RAFT) are adapted from their original codebases with small modifications. Please cite the original papers if you use these models.

The `original` configurations can be used to test the original checkpoints of the baseline models, so performance benchmarking can be done conveniently and with aligned standards. The `original` checkpoint files we provide are directly converted from the original weights using scripts such as `scripts/convert_checkpoint_from_original.py`. If you are an author of the original paper and find it inappropriate for us to redistribute your weights, please contact us and we will remove them.

Note that integrated test code for [NerNet+](https://github.com/Liu-haoyue/NER-Net) is also provided for benchmarking, although it cannot be retrained with the V2V framework since its input representation is not voxels. The ERAFT checkpoint provided is the MVSEC-20Hz version from [the original codebase](https://github.com/uzh-rpg/E-RAFT).

The `v2v` configurations are for training and testing the best of our models. We provide full runnable training code and checkpoints for them. We recommend using V2V-E2VID (`train_v2v_e2vid_10k.yaml`) for event-based video reconstruction, and V2V-ERAFT (`train_v2v_eraft_10k.yaml`) for optical flow estimation, since they achieve the best performance in our experiments. 

The `ablation` configurations demonstrate how to train ablation experiments, such as with the ESIM-280 dataset, fixed thresholds, filtered video lists, or degraded video inputs. We do not provide corresponding checkpoints. 

All listed checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1M6juFWeZ-bCkW9JNaSJzn0voLQ9LU-tg?usp=sharing). Put the downloaded checkpoints into `checkpoints/{experiment_name}` so they match the paths in `ckpt_paths/{experiment_name}.txt`.

## Dataset Preparation

### Testing

We provide data preparation scripts which cut sequences of the original HQF, EVAID, IJRR and MVSEC datasets, converting them to a unified h5 format. Specifically:

* HQF: Download from [HQF site](https://timostoff.github.io/20ecnn) and convert according to [these instructions](https://github.com/TimoStoff/event_cnn_minimal).

* EVAID: Download from [EVAID site](https://sites.google.com/view/eventaid-benchmark/home) and convert with `python scripts/evaid_to_h5.py`.

* IJRR (ECD): Download from [IJRR site](https://download.ifi.uzh.ch/rpg/web/data/E2VID/datasets/ECD_IJRR17/) and convert with `python scripts/ijrr_to_h5.py`.

* MVSEC: Download from [MVSEC site](https://daniilidis-group.github.io/mvsec/download/) and convert with `python scripts/mvsec_to_h5.py`.

If you would like to test on real data, I recommend converting it to the h5 format as well. The script `scripts/aedat4_to_h5.py` demonstrates how to convert Aedat4 files (output from DAVIS cameras) to the h5 format. A few of my own DAVIS346 captures are shared in the [EvBird](https://drive.google.com/drive/folders/1Fzu1h1XqaAVRwIRypujjUR66DQmHJHh2?usp=drive_link) directory, and testing can be performed using `config/test_evbird.yaml`.

### Training

The ESIM-280 training dataset (for baseline models) can be prepared by following the [E2VID++ instructions](https://github.com/TimoStoff/esim_config_generator). The script `scripts/esim_to_voxel.py` pre-stacks the events into interpolated or discrete voxel grids for acceleration.

To train with the V2V framework and the [WebVid dataset](https://github.com/m-bain/webvid), you would need to download the videos. The list of videos we used (a very small subset of WebVid) are provided in configuration files such as `config/webvid10000_unfiltered.txt`. 

## References

If you find this code useful, please cite our V2V paper:

```bibtex
@InProceedings{lou2025v2v,
  title={V2V: Scaling Event-Based Vision through Efficient Video-to-Voxel Simulation},
  author={Lou, Hanyue and Liang, Jinxiu and Teng, Minggui and Wang, Yi and Shi, Boxin},
  booktitle={arXiv preprint arXiv:2505.16797},
  year={2025}
}
```

When you use the original models, please also cite the corresponding papers:

```bibtex
@InProceedings{stoffregen2020e2videvflow,
  author        = {Timo, Stoffregen and Scheerlinck, Cedric and Scaramuzza, Davide and Drummond, Tom and Barnes, Nick and Kleeman, Lindsay and Mahony, Robert},
  title         = {Reducing the Sim-to-Real Gap for Event Cameras},
  booktitle     = {ECCV},
  year          = {2020},
}

@InProceedings{weng2021etnet,
    author    = {Weng, Wenming and Zhang, Yueyi and Xiong, Zhiwei},
    title     = {Event-based Video Reconstruction Using Transformer},
    booktitle = {ICCV},
    year      = {2021},
}

@Article{ercan2024hypere2vid,
  author = {Ercan, Burak and Eker, Onur and Sağlam, Canberk and Erdem, Aykut and Erdem, Erkut},
  title = {{HyperE2VID}: Improving Event-Based Video Reconstruction via Hypernetworks},
  journal = {TIP},
  year = {2024},
}

@Article{liu2025nernet,
  author={Liu, Haoyue and Xu, Jinghan and Peng, Shihan and Chang, Yi and Zhou, Hanyu and Duan, Yuxing and Zhu, Lin and Tian, Yonghong and Yan, Luxin},
  journal={TPAMI}, 
  title={{NER-Net+}: Seeing Motion at Nighttime with an Event Camera}, 
  year={2025},
}

@InProceedings{gehrig2021eraft,
    title={{E-RAFT}: Dense Optical Flow from Event Cameras},
    author={Gehrig, Mathias and Millhausler, Mario and Gehrig, Daniel and Scaramuzza, Davide},
    booktitle={3DV},
    year={2021},
}
```

If you meet with any issues, please **search the existing issues** of this repository and feel free to open a new issue if you cannot find a solution. I'll try to help when I'm available. :D