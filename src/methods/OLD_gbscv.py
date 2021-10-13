"""Generate graph-based cross-validation spatial folds"""
import os
import json
import time
from typing import Dict, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm
from src.scv import SCV


X_1DIM_COL = "X_1DIM"
X = "x"
Y = "y"


@dataclass
class GBSCV(SCV):
    """Generates the graph based spatial cross-validation folds
    Attributes
    ----------
        target_col: str
            The targer attribute column name
        adj_matrix: pd.Dataframe
            The adjacency matrix regarding the spatial objects in the data
    """

    target_col: str = "TARGET"
    fold_col: str = "FOLD_INDEX"
    lat_col: str = "[GEO]_LATITUDE"
    lon_col: str = "[GEO]_LONGITUDE"
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    _train_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    _test_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    _sill_target: Dict = field(default_factory=dict)
    _sill_reduced: Dict = field(default_factory=dict)

    def _get_index_train(self, index_test) -> List:
        """Return the train set indexes based on the test set indexes"""
        return [idx for idx in self.data.index if idx not in index_test]

    def _calculate_train_pca(self) -> np.array:
        """Return the PCA first component transformation on the traind data"""
        pca = PCA(n_components=1)
        train = self.data.drop(
            columns=[self.fold_col, self.target_col, self.lat_col, self.lon_col]
        )
        pca.fit(train)
        return pca.transform(train).flatten()

    def _convert_latlon_2_radians(self) -> pd.DataFrame:
        """Convert data latidude and longitude coordinates into radians"""
        self.data[self.lat_col] = self.data[self.lat_col].apply(np.radians)
        self.data[self.lon_col] = self.data[self.lon_col].apply(np.radians)

    def _split_data_test_train(self, test_data) -> pd.DataFrame:
        """Split the data into train and test set, based on a given teste set"""
        index_test = test_data.index
        index_train = self._get_index_train(index_test)
        self._test_data = self.data.loc[index_test].copy()
        self._train_data = self.data.loc[index_train].copy()

    def _calculate_removing_buffer_sill(self, fold_name, fold_data, global_var) -> Dict:
        """Calculate the sill for each fold to be used on the removing buffer process"""
        fold_target = fold_data[self.target_col]
        test_target = self._test_data[self.target_col]
        target_var = fold_target.append(test_target).var()
        self._sill_target[fold_name] = (target_var + global_var) / 2

    def _calculate_selection_buffer_sill(
        self, fold_name, fold_data, global_var
    ) -> Dict:
        """Calculate the sill for each fold to be used on the selection buffer process"""
        reduced_var = fold_data[X_1DIM_COL].append(self._test_data[X_1DIM_COL]).var()
        self._sill_reduced[fold_name] = (reduced_var + global_var) / 2
        max_var_train = max(self._sill_reduced, key=self._sill_reduced.get)
        for _ in self._sill_reduced:
            self._sill_reduced[_] = self._sill_reduced[max_var_train]

    def _initiate_buffers_sills(self) -> Dict:
        """Initialize and calculate the sills for the removing and selectiont procedures"""
        global_target_var = self.data[self.target_col].var()
        global_reduced_var = self.data[X_1DIM_COL].var()
        self._sill_target = {}
        self._sill_reduced = {}
        for fold_name, fold_data in self._train_data.groupby(by=self.fold_col):
            self._calculate_selection_buffer_sill(
                fold_name, fold_data, global_reduced_var
            )
            self._calculate_removing_buffer_sill(
                fold_name, fold_data, global_target_var
            )

    def _convert_adj_matrix_index_types(self) -> pd.DataFrame:
        """Convert adjacenty matrixy index and columns types to the same as in the data"""
        self.adj_matrix.index = self.adj_matrix.index.astype(self.data.index.dtype)
        self.adj_matrix.columns = self.adj_matrix.columns.astype(self.data.index.dtype)

    @staticmethod
    def _get_neighbors(indexes, adj_matrix) -> List:
        """Return the 1-degree neighborhood from a given sub-graph formed by indexes"""
        area_matrix = adj_matrix.loc[indexes]
        neighbors = area_matrix.sum(axis=0) > 0
        neighbors = neighbors[neighbors].index
        neighbors = [n for n in neighbors if n not in indexes]
        return neighbors

    def _calculate_longest_path(self) -> int:
        """Calculate the longest_path from a BFS tree taking the test set as root"""
        path_indexes = self._test_data.index.values.tolist()
        local_data_idx = (
            self._test_data.index.values.tolist()
            + self._train_data.index.values.tolist()
        )
        matrix = self.adj_matrix.loc[local_data_idx, local_data_idx]
        neighbors = self._get_neighbors(path_indexes, matrix)
        size_tree = 0
        while len(neighbors) > 0:
            size_tree += 1
            neighbors = self._get_neighbors(path_indexes, matrix)
            path_indexes = path_indexes + neighbors
        return size_tree

    def _calculate_similarity_matrix(self, fold_data, attribute) -> np.ndarray:
        """Calculate the similarity matrix between test set and a given training
        fold set based on a given attribute"""
        test_values = self._test_data[attribute].to_numpy()
        fold_values = fold_data[attribute].to_numpy()
        return np.subtract.outer(test_values, fold_values) ** 2

    def _calculate_geo_distance_weights(self, fold_data, weights) -> np.ndarray:
        """Calculate the distance weights to be used on the gamma calculation"""
        test_coord = self._test_data[[self.lat_col, self.lon_col]].to_numpy()
        fold_coord = fold_data[[self.lat_col, self.lon_col]].to_numpy()
        if weights == "spatial":
            return haversine_distances(test_coord, fold_coord)
        else:
            return np.full((test_coord.shape[0], fold_coord.shape[0]), 1)

    @staticmethod
    def _calculate_gamma(similarity, geo_weights) -> np.float64:
        """Calculate gamma or the semivariogram"""
        gamma_dist = np.multiply(similarity, geo_weights)
        gamma_dist = np.sum(gamma_dist, axis=1)
        geo_dist = np.sum(geo_weights, axis=1)
        sum_diff = gamma_dist.sum()
        sum_dist = geo_dist.sum()
        return sum_diff / (2 * sum_dist)

    def _calculate_gamma_by_fold(self, neighbors, attribute, weights) -> Dict:
        """Calculate the semivariogram by folds"""
        context_gamma = {}
        neighbors = [n for n in neighbors if n in self._train_data.index]
        neighbors_data = self._train_data.loc[neighbors]
        for fold, fold_data in neighbors_data.groupby(by=self.fold_col):
            similarity = self._calculate_similarity_matrix(fold_data, attribute)
            geo_weights = self._calculate_geo_distance_weights(fold_data, weights)
            gamma = self._calculate_gamma(similarity, geo_weights)
            context_gamma[fold] = {
                "gamma": gamma,
                "neighbors": fold_data.index.values.tolist(),
            }
        return context_gamma

    def _get_n_fold_neighbohood(self) -> int:
        """Get ne number of folds neighbors from the test set"""
        neighbors_idx = self._get_neighbors(self._test_data.index, self.adj_matrix)
        return len(self.data.loc[neighbors_idx].groupby(self.fold_col))

    @staticmethod
    def _calculate_exponent(size_tree, count_n, type_exp) -> np.float64:
        """Caclulate the decay exponent"""
        if type_exp == "log":
            return np.log(1 * size_tree - count_n) / np.log(1 * size_tree)
        else:
            return 1

    def _calculate_buffer(self, attribute, sill, kappa, weights, decay) -> List:
        """Calculate a buffer region"""
        # Initialize variables
        count_n = 0  # n-degree neighborhood counter
        growing = 1  # indicate wether the buffer still growing
        buffer = []  # containg the index of instaces buffered
        folds_in_buffer = []  # list of folds presented in the buffer
        # Get the size of the BFS tree with the test set as  root
        size_tree = self._calculate_longest_path()
        # Get the number of fold neighbors the test set has
        n_fold_neighbors = self._get_n_fold_neighbohood()
        # Start creating the buffer
        while growing:
            # Set growing to 0
            growing = 0
            # Get the instance indexes from te test set + the indexes buffer
            growing_graph_idx = self._test_data.index.values.tolist() + buffer
            # Get the neighbor
            neighbors = self._get_neighbors(growing_graph_idx, self.adj_matrix)
            # Calculate the semivariogram for each fold in the neighborhood
            context_gamma = self._calculate_gamma_by_fold(neighbors, attribute, weights)
            # Check for each fold in the neighborhood the semivariogram to decide
            # whether to add or not the instances into the buffer list
            for context_key, context in context_gamma.items():
                gamma = context["gamma"]
                exponent = self._calculate_exponent(size_tree, count_n, decay)
                sill_value = sill[context_key] * exponent
                n_contexts = len(set(folds_in_buffer + [context_key]))
                if gamma <= sill_value and n_contexts <= n_fold_neighbors * kappa:
                    folds_in_buffer.append(context_key)
                    growing = 1
                    buffer += context_gamma[context_key]["neighbors"]
            count_n += 1
        return buffer

    def _clean_data(self) -> pd.DataFrame:
        """Clean the dataset to present only attributes of interest"""
        cols_drop = [X_1DIM_COL, self.fold_col, self.lat_col, self.lon_col]
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

    def create_folds(
        self,
        run_selection=True,
        name_folds="folds",
        kappa=20,
        weights="non-spatial",
        decay="log",
    ):
        """Generate graph-based spatial folds"""
        # Create folder folds
        start_time = time.time()
        self._make_folders(["folds", name_folds])
        self.data[X_1DIM_COL] = self._calculate_train_pca()
        self._convert_latlon_2_radians()
        for fold_name, test_data in tqdm(self.data.groupby(by=self.fold_col), desc="Creating folds:"):
            # Cread fold folder
            self._mkdir(str(fold_name))
            # Initialize x , y and reduce
            self._split_data_test_train(test_data)
            # Calculate local sill
            self._initiate_buffers_sills()
            # Ensure indexes and columns compatibility
            self._convert_adj_matrix_index_types()
            # Calculate selection buffer
            if run_selection:
                selection_buffer = self._calculate_buffer(
                    X_1DIM_COL,
                    self._sill_reduced,
                    kappa=20,
                    weights=weights,
                    decay=decay,
                )
                selection_buffer = list(set(selection_buffer))
                self._train_data = self._train_data.loc[selection_buffer]
            # The train data is used to calcualte the buffer. Thus, the size tree,
            # and the gamma calculation will be influenced by the selection buffer.
            # Calculate removing buffer
            removing_buffer = self._calculate_buffer(
                self.target_col,
                self._sill_target,
                kappa=kappa,
                weights=weights,
                decay=decay,
            )
            removing_buffer = list(set(removing_buffer))
            self._train_data.drop(index=removing_buffer, inplace=True)
            # Save buffered data indexes
            self._save_buffered_indexes(removing_buffer)
            # Clean data
            self._clean_data()
            # Save data
            self._save_data()
            # Update cur dir
            self.cur_dir = os.path.join(self._get_root_path(), "folds", name_folds)
        end_time = time.time()
        print(f"Execution time: {end_time-start_time} seconds")
