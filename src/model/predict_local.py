"""Predict data process"""
import os
import re
from dataclasses import dataclass, field
from typing import List
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import r2_score
from scipy.special import softmax
from collections import Counter
import joblib
import pandas as pd
import numpy as np
import shap
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
    test_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: List = field(default_factory=list)

    def _read_test_data(self, json_path, data):
        """Read the training data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.test_data = data.loc[split_fold_idx["test"]].copy()

    def _read_train_data(self, json_path, data):
        """Read the training data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.train_data = data.loc[split_fold_idx["train"]].copy()

    def _selected_features_filtering(self, json_path, val):
        """Filter only the features selected"""

        selected_features = utils.load_json(json_path)
  
        if self.target_col not in selected_features["selected_features"]:          
            selected_features["selected_features"].append(self.target_col)
        self.test_data = self.test_data[selected_features["selected_features"]]
        self.train_data = self.train_data[selected_features["selected_features"]]
        if val:
            self.validation_data = self.validation_data[selected_features["selected_features"]]

    @staticmethod
    def load_model(filepath):
        """Load pickled models"""
        # return pickle.load(open(filepath, "rb"))
        return joblib.load(filepath)

    def _clean_train_data_col(self, val):
        clean_cols = [re.sub(r"\W+", "", col) for col in self.test_data.columns]
        self.test_data.columns = clean_cols
        self.train_data.columns = clean_cols
        if val:
            self.validation_data.columns = clean_cols
      

    def _split_data(self):
        """Split the data into explanatory and target features"""
        y_test = self.test_data[self.target_col]
        x_test = self.test_data.drop(columns=[self.target_col])
        return x_test, y_test
    
    def _split_data_train(self):
        """Split the data into explanatory and target features"""
        y_test = self.train_data[self.target_col]
        x_test = self.train_data.drop(columns=[self.target_col])
        return x_test, y_test
    
    def _split_data_validation(self):
        """Split the data into explanatory and target features"""
        y_val = self.validation_data[self.target_col]
        x_val = self.validation_data.drop(columns=[self.target_col])
        return x_val, y_val

    def _predict(self, model, val_error):
        """make prediction"""
        self._clean_train_data_col(False)
        x_test, _ = self._split_data()
        self.predictions = model.predict(x_test)
        return self.predictions

    def _predict_train(self, model):
        """make prediction"""
        self._clean_train_data_col(False)
        x_train, _ = self._split_data_train()
        self.predictions = model.predict(x_train)
        #error = np.random.uniform(low=-5, high=5, size=(len(self.predictions)))
        #error = np.random.normal(loc=0, scale=5, size=(len(self.predictions)))
        #self.predictions = self.predictions + error
        return self.predictions
    
    def _predict_validation(self, model):
        """make prediction"""
        self._clean_train_data_col(True)
        x_validation, _ = self._split_data_validation()
        self.predictions = model.predict(x_validation)
        return self.predictions
    
    def save_prediction(self, fold):
        """Save the model's prediction"""
        self.test_data[PRED_COL] = self.predictions
        self.test_data[GROUND_TRUTH_COL] = self.test_data[self.target_col].copy()
        pred_to_save = self.test_data[[PRED_COL, GROUND_TRUTH_COL]]
        pred_to_save.to_csv(os.path.join(self.cur_dir, f"{fold}.csv"))

    def _calculate_pca(self, data):
        """Return the PCA first component transformation on the traind data"""
        pca = PCA(n_components=1)
        data = data.drop(columns=[self.target_col])
        # For the IMCLA21 paper the PCA is executed only on the cennsus columns
        pca.fit(data)
        return pca.transform(data).flatten()

    def _calculate_similarity_matrix(self, pca_test, pca_train) -> np.ndarray:
        """Calculate the similarity matrix between test set and a given training
        fold set based on a given attribute"""
        return np.subtract.outer(pca_test, pca_train) ** 2

    @staticmethod
    def _calculate_gamma(similarity, weights) -> np.float64:
        """Calculate gamma or the semivariogram"""
        similarity = np.multiply(similarity, weights.to_numpy())
        gamma_dist = np.sum(similarity, axis=1)
        sum_diff = gamma_dist.sum()
        # sum_dist = similarity.size
        sum_dist = weights.to_numpy().sum()
        return sum_diff / (2 * sum_dist)
    
    
    def print_feature_importances_shap_values(self, shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))
        # Calculates the normalized version
        importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)[:7]}
        feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)[:7]}
        # Prints the feature importances
        #for k, v in feature_importances.items():
        #    print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")
        return feature_importances

    def run(self):
        """Runs the predicting process per fold"""
        data = pd.read_csv(os.path.join(self.root_path, "data.csv"))
        data.set_index(self.index_col, inplace=True)
        geo_weights = pd.read_csv(os.path.join(self.root_path, "normd_matrix.csv"))
        geo_weights.set_index(geo_weights.columns[0], inplace=True)
        self._make_folders(
            ["results", self.scv_method, "predictions", self.fs_method, self.ml_method,]
        )
        folds_path = os.path.join(self.root_path, "folds", self.scv_method)
        results_path = os.path.join(self.root_path, "results", self.scv_method)
        fs_path = os.path.join(results_path, "features_selected", self.fs_method)
        ml_path = os.path.join(
            results_path, "trained_models", self.fs_method, self.ml_method
        )
        folds_name = self._get_folders_in_dir(folds_path)
        folds_name.remove("53")
        folds_name.sort()
        folds_name = ["11"]
        shap_list = []
        for fold in tqdm(folds_name, desc="Predicting test set"):
            #print(fold)
            self._read_test_data(os.path.join(folds_path, fold), data)
            self._read_train_data(os.path.join(folds_path, fold), data)
            original_test = self.test_data
            original_train = self.train_data
            if "Local" in self.fs_method:
                self.train_data = self.train_data[self.train_data["INDEX_FOLDS"] != 53]
                context_list = self._get_files_in_dir(os.path.join(fs_path, fold))
                #context_list = [context for context in context_list if context not in ["12.json", "14.json", "16.json"] ]
                fold_pred_train = pd.DataFrame()
                fold_pred_test = pd.DataFrame()
                fold_pred_val = pd.DataFrame()
                fold_rsquared_val = {}
                dist_geo = {}
                # Find geo distance
                for fold_id, fold_train in self.train_data.groupby(by="INDEX_FOLDS"):
                    geo_weights_fold = geo_weights.loc[self.test_data.index, fold_train.index.astype("str")]
                    dist_geo[fold_id] = geo_weights_fold.mean().mean()
                dist_geo = dict(sorted(dist_geo.items(), key=lambda item: item[1]))
                validation_id = list(dist_geo.keys())[:7]
                context_list_validation = [f"{k}.json" for k in validation_id]
                #print(validation_id)
                #print(context_list)
                #context_list_validation = context_list.copy()
                #for id in validation_id:
                #    context_list_validation.remove(f"{id}.json")
                self.validation_data = self.train_data[self.train_data["INDEX_FOLDS"].isin(validation_id)] 
                original_val = self.validation_data
                for context in context_list_validation:
                    self._selected_features_filtering(
                        os.path.join(fs_path, fold, context)
                    , True)
                    model = self.load_model(
                        os.path.join(ml_path, fold, f"{context.split('.')[0]}.pkl")
                    )
                    prediction = self._predict_validation(model)
                    fold_pred_val[context.split(".")[0]] = prediction
                    fold_rsquared_val[context.split(".")[0]] = r2_score(self.validation_data[self.target_col], prediction)
                    self.test_data = original_test
                    self.train_data = original_train
                    self.validation_data = original_val
                #print(fold_rsquared_val)
                #val_mean_pred = fold_pred_val.mean(axis=1)
                #print(val_mean_pred)
                #print(self.validation_data[self.target_col])
                #val_mean_error = abs(self.validation_data[self.target_col].to_numpy() - val_mean_pred).mean()
                var_pred_per_fold = fold_pred_val.mean(axis=0)
            
                for col in fold_pred_val.columns:
                    fold_pred_val[col] = self.validation_data[self.target_col].to_numpy() - fold_pred_val[col]
                mean_error_per_fold = fold_pred_val.mean(axis=0)
                #print(mean_error_per_fold.sort_values())
                var_error_per_fold = fold_pred_val.var(axis=0)
                var_true_val = self.validation_data[self.target_col].mean()
         
                #print(fold, val_mean_error)
                context_list = [f"{k}.json" for k in list(dist_geo.keys())]
                
                for context in context_list:

                    self._selected_features_filtering(
                        os.path.join(fs_path, fold, context), False
                    )


                    model = self.load_model(
                        os.path.join(ml_path, fold, f"{context.split('.')[0]}.pkl")
                    )
                    fold_pred_train[context.split(".")[0]] = self._predict_train(model)
                    
                    
                    fold_pred_test[context.split(".")[0]] = self._predict(model, 1)
                    #if context.split(".")[0] == "29":
                    #    x_test, _ = self._split_data()
                    #    x_test.to_csv("11.csv")
                        
                    #    explainer = shap.Explainer(model.predict, x_test)
                        # Calculates the SHAP values - It takes some time
                    #    shap_values = explainer(x_test)
                    #    fi = self.print_feature_importances_shap_values(shap_values, x_test.columns.to_list())
                    #    print(fi)
                        #shap_list.append(fi)
                     
                  
                    self.test_data = original_test
                    self.train_data = original_train
                #if fold == "12":
                #    pd.set_option("display.max_rows", None, "display.max_columns", None)
                #    print(fold_pred_test)
                #    print(self.test_data[self.target_col].to_string())
                
                #print(fold)
                #print(max(mean_error_per_fold) - min(mean_error_per_fold))
                #print(mean_error_per_fold)
                #print(mean_error_per_fold.var())
                #print(mean_error_per_fold.mean())
                #for col in context_list:
                #    scala = 0
                #    if var_error_per_fold < .35 and abs(mean_error_per_fold.mean()) > .1:
                #        scala = 5
                #    fold_pred_test[col.split(".")[0]] = fold_pred_test[col.split(".")[0]] + scala*mean_error_per_fold[col.split(".")[0]]
                
                #for col in context_list:
                #    print(f"Fold {fold} - Context {col} - var_true  {var_true_val} - var_pred {var_pred_per_fold[col.split('.')[0]]}")
                #    if var_pred_per_fold[col.split(".")[0]] > 2*var_true_val:
                #        scala = abs(var_pred_per_fold - var_true_val)
                #        print(f"FOLD {fold} - Context {col} - Scala {scala}")
                #        fold_pred_test[col.split(".")[0]] = fold_pred_test[col.split(".")[0]] + scala*mean_error_per_fold[col.split(".")[0]]
                         
               # print(fold_pred_train)
                meta_model = LinearRegression()
                #cor_matrix = fold_pred_train.corr().abs()
                #upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
                #to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                print(fold_pred_train)
                meta_model.fit(fold_pred_train, self.train_data[self.target_col])
                #explainer = shap.Explainer(meta_model.predict, fold_pred_test)
                # Calculates the SHAP values - It takes some time
                #shap_values = explainer(fold_pred_test)
               
                #fi = self.print_feature_importances_shap_values(shap_values, fold_pred_train.columns.to_list())
                #print(fi)
                print("TEST================")
                print(fold_pred_test)
                self.predictions = meta_model.predict(fold_pred_test)
                print("PREDICTIONS================")
                print(self.predictions)
                
                
                #self.predictions = fold_pred_test.mean(axis=1).to_numpy()
                self.save_prediction(fold)
            else:
                self._selected_features_filtering(os.path.join(fs_path, f"{fold}.json"), False)
                model = self.load_model(os.path.join(ml_path, f"{fold}.pkl"))
                self._predict(model, 1)
            self.save_prediction(fold)

        #print(sum(shap_list)/len(shap_list))
        #print(folds_name)
