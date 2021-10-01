"""Generate raw data for census"""
import os
import json
import time
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.classes.data import Data


TRAIN = "train"
TEST = "test"
X = "x"
Y = "y"


@dataclass
class OPTMISTIC(Data):
    """Represents the Graph-Based Spatial Cross Validation.

    Attributes
    ----------

    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_col: str = "TARGET"
    fold_col: str = "FOLD_INDEX"
    lat_col: str = "[GEO]_LATITUDE"
    lon_col: str = "[GEO]_LONGITUDE"
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    _train_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    _test_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    _sill_target: np.float64 = None

    def _get_index_train(self, index_test):
        return [idx for idx in self.data.index if idx not in index_test]

    def _get_fold_x_y(self, index):
        return {
            X: self.data.loc[index].copy(),
            Y: self.data.loc[index, self.target_col].copy(),
        }

    def split_data_test_train(self, test_data):
        index_test = test_data.index
        index_train = self._get_index_train(index_test)
        self._test_data = self.data.loc[index_test]
        self._train_data = self.data.loc[index_train]

    def convert_adj_matrix_index_types(self):
        self.adj_matrix.index = self.adj_matrix.index.astype(self.data.index.dtype)
        self.adj_matrix.columns = self.adj_matrix.columns.astype(self.data.index.dtype)

    def clean_data(self):
        cols_drop = [self.fold_col, self.lat_col, self.lon_col]
        self._train_data.drop(columns=cols_drop, inplace=True)
        self._test_data.drop(columns=cols_drop, inplace=True)

    def save_data(self):
        self._train_data.reset_index(inplace=True)
        self._train_data.to_feather(os.path.join(self.cur_dir, "train.ftr"))
        self._test_data.reset_index(inplace=True)
        self._test_data.to_feather(os.path.join(self.cur_dir, "test.ftr"))

    def save_buffered_indexes(self):
        split_data = {
            "train": self._train_data.index.values.tolist(),
            "test": self._test_data.index.values.tolist(),
        }
        path_to_save = os.path.join(self.cur_dir, "split_data.json")
        with open(path_to_save, "w", encoding="utf-8") as file:
            json.dump(split_data, file)

    def create_folds(self) -> None:
        """Generate merged data"""
        # Create folder folds
        start_time = time.time()
        self._make_folders(["optmistic_folds"])
        for fold_name, test_data in tqdm(self.data.groupby(by=self.fold_col)):
            # Cread fold folder
            self._mkdir(str(fold_name))
            # Initialize x , y and reduce
            self.split_data_test_train(test_data)
            # Save buffered data indexes
            self.save_buffered_indexes()
            # Clean data
            self.clean_data()
            # Save data
            self.save_data()
            # Update cur dir
            self.cur_dir = os.path.join(self._get_root_path(), "optmistic_folds")
        end_time = time.time()
        print(f"Execution time: {end_time-start_time} seconds")
