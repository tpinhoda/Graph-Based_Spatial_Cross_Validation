
import subprocess
import time
import argparse
import os


if __name__ == "__main__":
    t1_start = time.process_time()
    root_path = "/home/tpinho/IJGIS/Datasets/QLD_Election_2019"
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--val_method",  # name on the CLI - drop the `--` for positional/required parameters
        nargs=1,  # 0 or more values expected => creates a list
        type=str,
        default="Optimistic",  # default if nothing is provided
    )
    CLI.add_argument(
        "--dataset_names",  # name on the CLI - drop the `--` for positional/required parameters
        nargs=1,  # 0 or more values expected => creates a list
        type=str,
        default="Original",  # default if nothing is provided
    )
    CLI.add_argument(
        "--folds",
        nargs="*",
        type=str,  # any type/callable can be used here
        default=[],
    )
    CLI.add_argument(
        "--list_contexts",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[],  # default if nothing is provided
    )

    index_col = "INDEX"
    target_col = "TARGET"
    fold_col = "INDEX_FOLDS"
    procs = []
    args = CLI.parse_args()
    print(args.dataset_names)
    for dataset_name in args.dataset_names:
        for fold in args.folds:
            cmd = f"python fs_cfs.py {args.val_method[0]} {root_path} {dataset_name} '{fold}' {index_col} {target_col} {fold_col}"
            print(cmd)
            procs.append(subprocess.Popen(cmd, shell=True))
    exit_codes = [p.wait() for p in procs]
    t1_stop = time.process_time()
    print(f"time -- {(t1_start-t1_stop)/60}")
