import os
import time
import json
import random
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    if rerun_mode:
        keys = sorted(data.keys())
    else:
        subset_path = "/kaggle/working/val_subset.json"
        if os.path.exists(subset_path):
            with open(subset_path, "r") as f:
                keys = json.load(f)
        else:
            n = int(os.getenv("VAL_SUBSET_N", "40"))
            rng = random.Random(1234)
            keys = sorted(rng.sample(sorted(data.keys()), n))
            with open(subset_path, "w") as f:
                json.dump(keys, f)

    for key in keys:
        queue.put(key)
    for _ in range(4):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
