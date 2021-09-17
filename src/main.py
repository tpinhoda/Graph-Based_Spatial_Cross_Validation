"""Main script"""
import os
from pathlib import Path
from src.classes.merge import Merge
from src import utils


def main():
    """Main function"""
    utils.initialize_coloredlog()
    utils.initialize_rich()
    utils.initialize_logging()
    # Project path
    project_dir = str(Path(__file__).resolve().parents[1])
    # Load enviromental variables
    env_var = utils.load_env_variables(project_dir)
    # Load paramenters
    params = utils.load_json(os.path.join(project_dir, "parameters", "parameters.json"))
    params["root_path"] = env_var["root_path"]
    params["census_filepath"] = env_var["census_filepath"]
    params["meshblock_filepath"] = env_var["meshblock_filepath"]
    params["other_filepath"] = env_var["other_filepath"]

    # Creates and run the meshblock processing pipeline
    merge = Merge(**params)
    merge.run()


if __name__ == "__main__":
    main()
