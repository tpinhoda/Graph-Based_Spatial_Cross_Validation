
"""Training data process"""
import os
import re
import math
from dataclasses import dataclass, field
import pickle
import joblib
import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
import pandas as pd
from tqdm import tqdm
from src.data import Data
import src.utils as utils

MAP_MODELS = {
"LGBM": lightgbm.LGBMRegressor,
"DT": DecisionTreeRegressor,
"SVM": SVR,
"KNN": KNeighborsRegressor,
"MLP": MLPRegressor,
"RF": RandomForestRegressor,
"Lasso": Lasso,
"OLS": LinearRegression,
"Ridge": Ridge,
"ElasticNet":ElasticNet
}


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

def _read_train_data( json_path, data):
    """Read the training data"""
    split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
    train_data = data.loc[split_fold_idx["train"]].copy()

def _selected_features_filtering( json_path):
    """Filter only the features selected"""
    selected_features = utils.load_json(json_path)
    selected_features["selected_features"].append(target_col)
    train_data = train_data[selected_features["selected_features"]]

def _get_model( params):
    """Get the models by name"""
    if ml_method == "KNN":
        return MAP_MODELS[ml_method](
            n_neighbors=math.floor(math.sqrt(train_data.shape[0]))
        )
    if ml_method == "MLP":
        return MAP_MODELS[ml_method](
            (math.floor(train_data.shape[1] / 2),),
            random_state=1,
            max_iter=50000,
            learning_rate_init=0.001,
            learning_rate="invscaling",
            shuffle=False,
            early_stopping=False,
            batch_size=100,
            tol=1e-2,
            activation="relu",
            solver="adam",
        )
    if ml_method == "RF":
        return MAP_MODELS[ml_method](n_estimators=200, random_state=1)
    if ml_method == "DT":
        return MAP_MODELS[ml_method]( random_state=1)
    if ml_method == "Lasso":
        return MAP_MODELS[ml_method](alpha=0.001, random_state=1)
    if ml_method == "OLS":
        return MAP_MODELS[ml_method]()
    if ml_method == "Ridge":
        return MAP_MODELS[ml_method](alpha=0.001)
    if ml_method == "ElasticNet":
        return MAP_MODELS[ml_method](alpha=0.001)
    if ml_method == "SVM":
        return MAP_MODELS[ml_method]()
    return MAP_MODELS[ml_method](*params)

def _split_data(self):
    """Split the data into explanatory and target features"""
    _clean_train_data_col()
    y_train = train_data[target_col]
    x_train = train_data.drop(columns=[target_col])
    return x_train, y_train

def _clean_train_data_col(self):
    clean_cols = [re.sub(r"\W+", "", col) for col in train_data.columns]
    train_data.columns = clean_cols

def _fit( model):
    """Fit the model"""
    x_train, y_train = _split_data()
    return model.fit(x_train, y_train)

def save_model( model, fold):
    """Save the model using picke"""
    joblib.dump(model, open(os.path.join(cur_dir, f"{fold}.pkl"), "wb"), compress=9)

def main(self):
    """Runs the training process per fold"""
    data = pd.read_csv(os.path.join(root_path, "data.csv"))
    data.set_index(index_col, inplace=True)

    _make_folders(
        [
            "results",
            scv_method,
            "trained_models",
            fs_method,
            ml_method,
        ]
    )
    folds_path = os.path.join(root_path, "folds", scv_method)
    fs_path = os.path.join(
        root_path,
        "results",
        scv_method,
        "features_selected",
        fs_method,
    )
    folds_name = _get_folders_in_dir(folds_path)
    if any(
        method in ml_method
        for method in [
            "MLP",
            "RF",
            "DT",
            "Lasso",
            "KNN",
            "LGBM",
            "OLS",
            "SVM",
            "Ridge",
            "ElasticNet"
        ]
    ):
        ml_method = ml_method.split("_", maxsplit=1)[0]
    for fold in tqdm(folds_name, desc="Training model"):
        params = {}
        _read_train_data(os.path.join(folds_path, fold), data)
        _selected_features_filtering(os.path.join(fs_path, f"{fold}.json"))
        model = _get_model(params=params)
        model = _fit(model)
        save_model(model, fold)
        
if __name__ == "__main__":
    root_path = sys.argv[1]
    dataset = sys.argv[2]
    fs_method = sys.argv[3]
    index_col = sys.argv[4]
    fold_col = sys.argv[5]
    target_col = sys.argv[6]
    ml_method = sys.argv[7]
    main(root_path, dataset, fs_method, index_col, fold_col, target_col, ml_method)






