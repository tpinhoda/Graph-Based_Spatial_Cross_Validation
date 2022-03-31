import sys
import subprocess
import time
from tqdm import tqdm



if __name__ == "__main__":
    t1_start = time.process_time() 
    root_path = "/exp/tpinho/Datasets/"
    val_method = sys.argv[2]
    dataset_names = [sys.argv[1]]
    index_col = "INDEX"
    target_col = "TARGET"
    fold_col = "INDEX_FOLDS"
    #folds = [51]
    folds = [11, 12, 13, 14, 15, 16, 17, 21, 22]
    #folds = [23, 24, 25, 26, 27, 28, 29, 31, 32]
    #folds = [33, 35, 41, 42, 43, 50, 51, 52, 53]
    
    procs = []
    for dataset_name in dataset_names:
        for fold in folds:
            cmd = f"python fs_cfs.py {val_method} {root_path} {dataset_name} {fold} {index_col} {target_col} {fold_col}"
            procs.append(
                subprocess.Popen(cmd, shell=True)
            )
    exit_codes = [p.wait() for p in procs]
    t1_stop = time.process_time()
    print(f"time -- {(t1_start-t1_stop)/60}")