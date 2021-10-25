"""Predict data process"""
import os
from dataclasses import dataclass, field
from typing import List
import pickle
import pandas as pd
from tqdm import tqdm
from src.data import Data
import src.utils as utils

PRED_COL = "PREDICTIONS"
GROUND_TRUTH_COL = "GROUND_TRUTH"


@dataclass
class Predict(Data):
    """Represents the predict data process.

     Attributes
    ----------
        ml_method:str
            The machine learning method name
        fs_method:str
            The feature selection method name
        scv_method:str
            The spatial cross-validation method name
        index_col: str
            The dataset´s index column name
        target_col: str
            The target column name
        root_path : str
            Root path
    """

    ml_method: str = "LGBM"
    fs_method: str = "CFS"
    scv_method: str = "gbscv"
    index_col: str = "INDEX"
    target_col: str = "TARGET"
    _test_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    _predictions: List = field(default_factory=list)

    def _read_test_data(self, data_path):
        """Read the training data"""
        self._test_data = pd.read_feather(os.path.join(data_path, "test.ftr"))
        self._test_data.set_index(self.index_col, inplace=True)

    def _selected_features_filtering(self, json_path):
        """Filter only the features selected"""
        selected_features = utils.load_json(json_path)
        selected_features["selected_features"].append(self.target_col)
        self._test_data = self._test_data[selected_features["selected_features"]]

    @staticmethod
    def load_model(filepath):
        """Load pickled models"""
        return pickle.load(open(filepath, "rb"))

    def _split_data(self):
        """Split the data into explanatory and target features"""
        y_test = self._test_data[self.target_col]
        x_test = self._test_data.drop(columns=[self.target_col])
        return x_test, y_test

    def _predict(self, model):
        """make prediction"""
        x_test, _ = self._split_data()
        self._predictions = model.predict(x_test)

    def save_prediction(self, fold):
        """Save the model's prediction"""
        self._test_data[PRED_COL] = self._predictions
        self._test_data[GROUND_TRUTH_COL] = self._test_data[self.target_col]
        pred_to_save = self._test_data[[PRED_COL, GROUND_TRUTH_COL]]
        pred_to_save.to_csv(os.path.join(self._cur_dir, f"{fold}.csv"))

    def run(self):
        """Runs the predicting process per fold"""
        self._make_folders(
            [
                "results",
                self.scv_method,
                "predictions",
                self.fs_method,
                self.ml_method,
            ]
        )
        folds_path = os.path.join(self.root_path, "folds", self.scv_method)
        results_path = os.path.join(self.root_path, "results", self.scv_method)
        fs_path = os.path.join(results_path, "features_selected", self.fs_method)
        ml_path = os.path.join(
            results_path, "trained_models", self.fs_method, self.ml_method
        )
        folds_name = self._get_folders_in_dir(folds_path)
        for fold in tqdm(folds_name, desc="Predicting test set"):
            self._read_test_data(os.path.join(folds_path, fold))
            self._selected_features_filtering(os.path.join(fs_path, f"{fold}.json"))
            model = self.load_model(os.path.join(ml_path, f"{fold}.pkl"))
            self._predict(model)
            self.save_prediction(fold)
