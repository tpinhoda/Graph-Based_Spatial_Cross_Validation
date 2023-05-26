echo CrossValidation
python global_run_subprocess.py --dataset_names Original --val_method CrossValidation --fold 1 2 3 4 5 6 7 8 9 10 --list_contexts -100
echo Optimistic
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo TraditionalSCV
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.1 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.2 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.3 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.4 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.5 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.6 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.7 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.8 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.9 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
echo Australia_election
#python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_1.0 --fold ACT NSW NT QLD SA TAS VIC WA --list_contexts -100
