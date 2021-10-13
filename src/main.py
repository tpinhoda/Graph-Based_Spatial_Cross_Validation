"""Main script"""
import os
from pathlib import Path
from src.methods.optimistic import OPTMISTIC
from src.methods.ultra_coservative import ULTRACONSERVATIVE
from src.methods.gbscv import GBSCV
import pandas as pd
from src import utils
import rich

rich.traceback.install()


def main():
    """Main function"""
    utils.initialize_coloredlog()
    utils.initialize_rich()
    utils.initialize_logging()
    # Project path
    project_dir = str(Path(__file__).resolve().parents[1])
    # Load enviromental variables
    env_var = utils.load_env_variables(project_dir)
    # Set parameters
    index = "INDEX"
    index_folds = "INDEX_FOLDS"
    target = "TARGET"
    lat_col = "GEO_y"
    lon_col = "GEO_x"
    # Load data
    path = os.path.join(env_var["root_path"], "data.csv")
    data = pd.read_csv(path, index_col=index, low_memory=False)
    adj_matrix = pd.read_csv(
        os.path.join(env_var["root_path"], "queen_matrix.csv"), low_memory=False
    )
    # Remove lat lon cols
    data.drop(columns=[lat_col, lon_col], inplace=True)
    # Set index cols
    adj_matrix.set_index(adj_matrix.columns[0], inplace=True)
    # Creates and run the meshblock processing pipeline
    #gbscv = GBSCV(
    #    data=data,
    #    fold_col=index_folds,
    #    target_col=target,
    #    adj_matrix=adj_matrix,
    #    root_path=env_var["root_path"],
    #)
    #gbscv.create_folds(
    #    run_selection=True,
    #    name_folds="selection_removing_buffer_folds",
    #    kappa=20,
    #)
    
    #optmistic = OPTMISTIC(
    #    data=data,
    #    fold_col=index_folds,
    #    root_path=env_var["root_path"],
    #)
    #optmistic.create_folds(
    #    name_folds="optmistic_folds",
    #)
    conservative = ULTRACONSERVATIVE(
        data=data,
        fold_col=index_folds,
        target_col=target,
        adj_matrix=adj_matrix,
        root_path=env_var["root_path"],
    )
    conservative.create_folds(
        name_folds="ultra_conservative_folds",
    )
    

if __name__ == "__main__":
    main()
