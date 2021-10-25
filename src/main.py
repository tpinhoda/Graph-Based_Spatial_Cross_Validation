"""Main script"""
import os
from pathlib import Path
import pandas as pd
from src import utils
from src.pipeline import Pipeline

SWITCHERS = {
    "scv": True,
    "fs": True,
    "train": True,
    "predict": True,
    "evaluate": True,
    "visualization": False,
}


def main():
    """Main function"""
    utils.initialize_coloredlog()
    utils.initialize_rich_tracerback()
    utils.initialize_logging()
    # Project path
    project_dir = str(Path(__file__).resolve().parents[1])
    # Load enviromental variables
    env_var = utils.load_env_variables(project_dir)
    # Set parameters
    index = "INDEX"
    # Load data
    path = os.path.join(env_var["root_path"], "data.csv")
    data = pd.read_csv(path, index_col=index, low_memory=False)

    adj_matrix = pd.read_csv(
        os.path.join(env_var["root_path"], "queen_matrix.csv"), low_memory=False
    )
    # Set index cols
    adj_matrix.set_index(adj_matrix.columns[0], inplace=True)
    pipeline = Pipeline(
        root_path=env_var["root_path"],
        data=data,
        adj_matrix=adj_matrix,
        index_col=index,
        fold_col="INDEX_FOLDS",
        target_col="TARGET",
        scv_method="UltraConservative",
        run_selection=False,
        kappa=20,
        fs_method="CFS",
        ml_method="LGBM",
        paper=True,
        switchers=SWITCHERS,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
