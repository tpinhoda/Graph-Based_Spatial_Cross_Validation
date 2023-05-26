echo CrossValidation
python global_run_subprocess.py --Original --val_method CrossValidation --fold 1 2 3 4 5 6 7 8 9 10 --list_contexts -100
echo Optimistic
python global_run_subprocess.py --Original --val_method Optimistic --fold 11 12 13 14 15 16 17 21 --list_contexts -100
python global_run_subprocess.py --Original --val_method Optimistic --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --Original --val_method Optimistic --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo TraditionalSCV
echo Brazil_election
python global_run_subprocess.py --Original --val_method TraditionalSCV --fold 11 12 13 14 15 16 17 21 --list_contexts -100
python global_run_subprocess.py --Original --val_method TraditionalSCV --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --Original --val_method TraditionalSCV --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
