import glob
import os
import sys

exp_list = sorted(glob.glob("checkpoints/*"))
exp_list = [os.path.basename(exp) for exp in exp_list]
exp_list = [exp for exp in exp_list if exp[2] != "5"]

print(exp_list)

for exp_name in exp_list:
    ckpt_path_file = f"ckpt_paths/{exp_name}.txt"
    try:
        with open(ckpt_path_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != ""]
            last_epoch = lines[-1].split("/")[-1]
            print(exp_name, last_epoch)

        all_ckpts = sorted(glob.glob(f"checkpoints/{exp_name}/*.pth"))
        for ac in all_ckpts:
            if os.path.basename(ac) != last_epoch:
                print(f"Removing {ac}")
                os.remove(ac)
    except Exception as e:
        print(e)