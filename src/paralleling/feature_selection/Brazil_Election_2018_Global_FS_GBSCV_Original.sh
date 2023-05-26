echo CrossValidation
python global_run_subprocess.py --Original --val_method CrossValidation --fold 1 2 3 4 5 6 7 8 9 10 --list_contexts -100
echo Optimistic
python global_run_subprocess.py --Original --val_method Optimistic --fold 11 12 13 14 15 16 17 21 --list_contexts -100
python global_run_subprocess.py --Original --val_method Optimistic --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --Original --val_method Optimistic --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo TraditionalSCV
python global_run_subprocess.py --Original --val_method TraditionalSCV --fold 11 12 13 14 15 16 17 21 --list_contexts -100
python global_run_subprocess.py --Original --val_method TraditionalSCV --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --Original --val_method TraditionalSCV --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.1 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.1 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.1 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.2 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.2 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.2 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.3 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.3 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.3 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.4 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.4 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.4 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.5 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.5 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.5 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.6 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.6 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.6 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.7 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.7 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.7 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.8 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.8 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.8 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.9 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.9 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.9 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
echo Brazil_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_1.0 --fold 11 12 13 14 15 16 17 21 22 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_1.0 --fold 23 24 25 26 27 28 29 31 52 --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_1.0 --fold 32 33 35 41 42 43 50 51 53  --list_contexts -100
