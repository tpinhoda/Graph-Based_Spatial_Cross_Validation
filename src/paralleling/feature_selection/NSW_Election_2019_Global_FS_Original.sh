echo CV
python global_run_subprocess.py --dataset_names Original --val_method CrossValidation --fold 1 2 3 4 5 6 7 8 9 10 --list_contexts -100
echo Optimistic
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Banks Barton Bennelong Berowra Blaxland Bradfield Calare Chifley  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Cook Cowper Cunningham Dobell Eden-Monaro Farrer Fowler Gilmore Grayndler  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Greenway Hughes Hume Hunter "Kingsford Smith" Lindsay Lyne Macarthur  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Mackellar Macquarie McMahon Mitchell "New England" Newcastle "North Sydney" Chifley  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Page Parkes Parramatta Paterson Reid Richmond Riverina Robertson  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method Optimistic --fold Shortland Sydney Warringah Watson Wentworth Werriwa Whitlam  --list_contexts -100
echo TraditionalSCV
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Banks Barton Bennelong Berowra Blaxland Bradfield Calare Chifley  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Cook Cowper Cunningham Dobell Eden-Monaro Farrer Fowler Gilmore Grayndler  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Greenway Hughes Hume Hunter "Kingsford Smith" Lindsay Lyne Macarthur  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Mackellar Macquarie McMahon Mitchell "New England" Newcastle "North Sydney" Chifley  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Page Parkes Parramatta Paterson Reid Richmond Riverina Robertson  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method TraditionalSCV --fold Shortland Sydney Warringah Watson Wentworth Werriwa Whitlam  --list_contexts -100
echo RegGBSCV_R_Kappa_0.0
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Banks Barton Bennelong Berowra Blaxland Bradfield Calare Chifley  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Cook Cowper Cunningham Dobell Eden-Monaro Farrer Fowler Gilmore Grayndler  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Greenway Hughes Hume Hunter "Kingsford Smith" Lindsay Lyne Macarthur  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Mackellar Macquarie McMahon Mitchell "New England" Newcastle "North Sydney" Chifley  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Page Parkes Parramatta Paterson Reid Richmond Riverina Robertson  --list_contexts -100
python global_run_subprocess.py --dataset_names Original --val_method RegGBSCV_R_Kappa_0.0 --fold Shortland Sydney Warringah Watson Wentworth Werriwa Whitlam  --list_contexts -100
