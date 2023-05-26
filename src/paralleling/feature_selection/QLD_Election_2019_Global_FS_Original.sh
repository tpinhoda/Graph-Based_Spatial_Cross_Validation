echo CV
#python global_run_subprocess.py --dataset_names Original --val_method CrossValidation --fold 1 2 3 4 5 6 7 8 9 10 --list_contexts -100
echo Optimistic
#python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Blair Bonner Bowman Brisbane Capricornia Dawson Dickson Fadden Fairfax Ryan  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Leichhardt Fisher Flynn Forde Griffith Groom Herbert Hinkler Kennedy "Wide Bay"  --list_contexts -100
#python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Lilley Longman Maranoa McPherson Moncrieff Moreton Oxley Petrie Rankin Wright  --list_contexts -100
echo TraditionalSCV
#python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Blair Bonner Bowman Brisbane Capricornia Dawson Dickson Fadden Fairfax Ryan  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Leichhardt Fisher Flynn Forde Griffith Groom Herbert Hinkler Kennedy "Wide Bay"  --list_contexts -100
#python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Lilley Longman Maranoa McPherson Moncrieff Moreton Oxley Petrie Rankin Wright  --list_contexts -100
echo RegGBSCV_R_Kappa_0
#python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Blair Bonner Bowman Brisbane Capricornia Dawson Dickson Fadden Fairfax Ryan  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Leichhardt Fisher Flynn Forde Griffith Groom Herbert Hinkler Kennedy "Wide Bay"  --list_contexts -100
#python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Lilley Longman Maranoa McPherson Moncrieff Moreton Oxley Petrie Rankin Wright  --list_contexts -100
