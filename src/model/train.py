"""Training data process"""
import os
import re
from dataclasses import dataclass, field
import pickle
import lightgbm
import pandas as pd
from tqdm import tqdm
from src.data import Data
import src.utils as utils

MAP_MODELS = {"LGBM": lightgbm.LGBMRegressor}


@dataclass
class Train(Data):
    """Represents the training data process.

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
    train_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _read_train_data(self, data_path):
        """Read the training data"""
        self.train_data = pd.read_feather(os.path.join(data_path, "train.ftr"))
        self.train_data.set_index(self.index_col, inplace=True)

    def _selected_features_filtering(self, json_path):
        """Filter only the features selected"""
        selected_features = utils.load_json(json_path)
        selected_features["selected_features"].append(self.target_col)
        self.train_data = self.train_data[selected_features["selected_features"]]

    def _get_model(self, params):
        """Get the models by name"""
        return MAP_MODELS[self.ml_method](*params)

    def _split_data(self):
        """Split the data into explanatory and target features"""
        self._clean_train_data_col()
        y_train = self.train_data[self.target_col]
        x_train = self.train_data.drop(columns=[self.target_col])
        return x_train, y_train
    
    def _clean_train_data_col(self):
        clean_cols = [re.sub(r'\W+','', col) for col in self.train_data.columns]
        self.train_data.columns = clean_cols
        
    def _fit(self, model):
        """Fit the model"""
        x_train, y_train = self._split_data()
        return model.fit(x_train, y_train)

    def save_model(self, model, fold):
        """Save the model using picke"""
        pickle.dump(model, open(os.path.join(self.cur_dir, f"{fold}.pkl"), "wb"))

    def run(self):
        """Runs the training process per fold"""
        self._make_folders(
            [
                "results",
                self.scv_method,
                "trained_models",
                self.fs_method,
                self.ml_method,
            ]
        )
        folds_path = os.path.join(self.root_path, "folds", self.scv_method)
        fs_path = os.path.join(
            self.root_path,
            "results",
            self.scv_method,
            "features_selected",
            self.fs_method,
        )
        folds_name = self._get_folders_in_dir(folds_path)
        for fold in tqdm(folds_name, desc="Training model"):
            self._read_train_data(os.path.join(folds_path, fold))
            self._selected_features_filtering(os.path.join(fs_path, f"{fold}.json"))
            model = self._get_model(params={})
            model = self._fit(model)
            self.save_model(model, fold)
