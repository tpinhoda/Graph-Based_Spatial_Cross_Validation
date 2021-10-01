"""Main script"""
import os
from pathlib import Path
from src.classes.optimistic import OPTMISTIC
from src.classes.ultra_coservative import ULTRACONSERVATIVE
from src.classes.gbscv import GBSCV
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
    adj_matrix.set_index(adj_matrix.columns[0], inplace=True)
    # Creates and run the meshblock processing pipeline
    gbscv = GBSCV(
        data=data,
        target_col=target,
        fold_col=index_folds,
        lat_col=lat_col,
        lon_col=lon_col,
        adj_matrix=adj_matrix,
        root_path=env_var["root_path"],
    )
    gbscv.create_folds(
        run_selection=1,
        name_folds="selection_removing_buffer_folds",
        kappa=20,
        weights="non-spatial",
        decay="log",
    )
    # gbscv.create_folds()


if __name__ == "__main__":
    main()
