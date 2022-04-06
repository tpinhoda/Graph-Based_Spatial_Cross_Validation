import sys
import subprocess
import time
from tqdm import tqdm



if __name__ == "__main__":
    t1_start = time.process_time() 
    root_path = "/exp/tpinho/Datasets/"
    original_ds = "Brazil_Election_2018"
    root_path = "/exp/tpinho/Datasets"
    fs_method = "CFS"
    index_col = "INDEX"
    target_col = "TARGET"
    fold_col = "INDEX_FOLDS"
    brazil_datasets = ["Brazil_Election_2018_Sampled_dec0.3_prob0.1",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.2",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.3",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.4",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.5",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.6",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.7",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.8",
                   "Brazil_Election_2018_Sampled_dec0.3_prob0.9"]


    
    procs = []
    for dataset_name in brazil_datasets:
        cmd = f"python out_sampled_train_pred.py {root_path} {dataset_name} {original_ds} {fs_method} {index_col} {fold_col} {target_col}"
        procs.append(
            subprocess.Popen(cmd, shell=True)
        )
    exit_codes = [p.wait() for p in procs]
    t1_stop = time.process_time()
    print(f"time -- {(t1_start-t1_stop)/60}")