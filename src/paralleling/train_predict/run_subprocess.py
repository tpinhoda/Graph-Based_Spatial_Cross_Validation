import sys
import subprocess
import time
from tqdm import tqdm



if __name__ == "__main__":
    t1_start = time.process_time() 
    root_path = "/exp/tpinho/Datasets/"
    val_method = sys.argv[1]
    kappa = sys.argv[2]
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
        if val_method == "TraditionalSCV":
            cmd = f"python traditionalscv.py {root_path} {dataset_name} {fs_method} {index_col} {fold_col} {target_col}"
            procs.append(
                subprocess.Popen(cmd, shell=True)
            )
        if val_method == "Optimistic":
            cmd = f"python optimistic.py {root_path} {dataset_name} {fs_method} {index_col} {fold_col} {target_col}"
            procs.append(
                subprocess.Popen(cmd, shell=True)
            )
        if val_method == "RegGBSCV":
            cmd = f"python reggbscv.py {root_path} {dataset_name} {fs_method} {index_col} {fold_col} {target_col} {kappa}"
            procs.append(
                subprocess.Popen(cmd, shell=True)
            )
    exit_codes = [p.wait() for p in procs]
    t1_stop = time.process_time()
    print(f"time -- {(t1_start-t1_stop)/60}")