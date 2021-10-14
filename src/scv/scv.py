"""Generate spatial folds"""
from abc import ABC, abstractmethod
import os
import json
from dataclasses import dataclass, field
from typing import List
import pandas as pd
from src.data import Data


@dataclass
class SCV(Data, ABC):
    """Represents the Spatial Cross Validation.

     Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_col: str = "FOLD_INDEX"
    _train_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    _test_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _get_index_train(self, index_test) -> List:
        """Return the train set indexes based on the test set indexes"""
        return [idx for idx in self.data.index if idx not in index_test]

    def _split_data_test_train(self, test_data) -> pd.DataFrame:
        """Split the data into train and test set, based on a given teste set"""
        index_test = test_data.index
        index_train = self._get_index_train(index_test)
        self._test_data = self.data.loc[index_test].copy()
        self._train_data = self.data.loc[index_train].copy()

    def _clean_data(self, cols_drop: List):
        """Clean the dataset to present only attributes of interest"""
        self._train_data.drop(columns=cols_drop, inplace=True)
        self._test_data.drop(columns=cols_drop, inplace=True)

    def _save_data(self):
        """Save the train and test set using feather"""
        self._train_data.reset_index(inplace=True)
        self._train_data.to_feather(os.path.join(self.cur_dir, "train.ftr"))
        self._test_data.reset_index(inplace=True)
        self._test_data.to_feather(os.path.join(self.cur_dir, "test.ftr"))

    def _save_buffered_indexes(self, removing_buffer):
        """Save the indexes of the buffers"""
        train_test_idx = (
            self._train_data.index.values.tolist()
            + self._test_data.index.values.tolist()
        )
        discarded_idx = [
            _ for _ in self.data.index if _ not in train_test_idx + removing_buffer
        ]
        split_data = {
            "train": self._train_data.index.values.tolist(),
            "test": self._test_data.index.values.tolist(),
            "removing_buffer": removing_buffer,
            "discarded": discarded_idx,
        }
        path_to_save = os.path.join(self.cur_dir, "split_data.json")
        with open(path_to_save, "w", encoding="utf-8") as file:
            json.dump(split_data, file, indent=4)

    @abstractmethod
    def create_folds(self, run_selection=True, name_folds="gbscv", kappa=20):
        """Generate graph-based spatial folds"""
