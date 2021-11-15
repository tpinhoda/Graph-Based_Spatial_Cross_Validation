"""Generate graph-based cross-validation spatial folds"""
import os
import time
from typing import Dict, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.scv.scv import SpatialCV


X_1DIM_COL = "X_1DIM"
SRBUFFER = "RegGBSCV_SR"
RBUFFER = "RegGBSCV_R"


@dataclass
class RegGraphBasedSCV(SpatialCV):
    """Generates the Graph Based Spatial Cross-Validation folds
    Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
        target_col: str
            The targer attribute column name
        adj_matrix: pd.Dataframe
            The adjacency matrix regarding the spatial objects in the data
        paper: bool
            Whether to run experiments according to ICMLA21 paper
        root_path : str
            Root path
    """

    kappa: int = 20
    run_selection: bool = False
    target_col: str = "TARGET"
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper: bool = False
    type_graph: str = "Sparse"
    sill_target: Dict = field(default_factory=dict)
    sill_reduced: Dict = field(default_factory=dict)
    w_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def _init_fields(self):
        if self.type_graph == "Sparse": 
            self.w_matrix = pd.DataFrame(index=self.adj_matrix.index, columns=self.adj_matrix.columns)
            self.w_matrix.fillna(1, inplace=True)
        else:
            self.w_matrix = self.adj_matrix
        self.sill_target = {}
        self.sill_reduced = {}
        
    def _calculate_train_pca(self) -> np.array:
        """Return the PCA first component transformation on the traind data"""
        pca = PCA(n_components=1)
        train = self.data.drop(columns=[self.fold_col, self.target_col])
        # For the IMCLA21 paper the PCA is executed only on the cennsus columns
        if self.paper:
            cols = [c for c in train.columns if "CENSUS" in c]
            train = train[cols]
        pca.fit(train)
        return pca.transform(train).flatten()

    def _calculate_removing_buffer_sill(self, fold_name, fold_data, global_var) -> Dict:
        """Calculate the sill for each fold to be used on the removing buffer process"""
        fold_target = fold_data[self.target_col]
        test_target = self.test_data[self.target_col]
        target_var = fold_target.append(test_target).var()
        self.sill_target[fold_name] = (target_var + global_var) / 2

    def _calculate_selection_buffer_sill(
        self, fold_name, fold_data, global_var
    ) -> Dict:
        """Calculate the sill for each fold to be used on the selection buffer process"""
        reduced_var = fold_data[X_1DIM_COL].append(self.test_data[X_1DIM_COL]).var()
        self.sill_reduced[fold_name] = (reduced_var + global_var) / 2
        max_var_train = max(self.sill_reduced, key=self.sill_reduced.get)
        for _ in self.sill_reduced:
            self.sill_reduced[_] = self.sill_reduced[max_var_train]

    def _initiate_buffers_sills(self) -> Dict:
        """Initialize and calculate the sills for the removing and selectiont procedures"""
        global_target_var = self.data[self.target_col].var()
        global_reduced_var = self.data[X_1DIM_COL].var()
        self.sill_target = {}
        self.sill_reduced = {}
        for fold_name, fold_data in self.train_data.groupby(by=self.fold_col):
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
        self.w_matrix.index = self.w_matrix.index.astype(self.data.index.dtype)
        self.w_matrix.columns = self.w_matrix.columns.astype(self.data.index.dtype) 

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
        path_indexes = self.test_data.index.values.tolist()
        local_data_idx = (
            self.test_data.index.values.tolist()
            + self.train_data.index.values.tolist()
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
        test_values = self.test_data[attribute].to_numpy()
        fold_values = fold_data[attribute].to_numpy()
        return np.subtract.outer(test_values, fold_values) ** 2

    @staticmethod
    def _calculate_gamma(similarity, geo_weights) -> np.float64:
        """Calculate gamma or the semivariogram"""
        gamma_dist = np.multiply(similarity, geo_weights)
        gamma_dist = np.sum(gamma_dist, axis=1)
        geo_dist = np.sum(geo_weights, axis=1)
        sum_diff = gamma_dist.sum()
        sum_dist = geo_dist.sum()
        return sum_diff / (2 * sum_dist)
    
    def _get_neighbors_weights(self, fold_data):
   #     fold_data_nodes = [n for n in fold_data.index if n in self.w_matrix.columns]
        return self.w_matrix.loc[self.test_data.index, fold_data.index]

    def _calculate_gamma_by_fold(self, neighbors, attribute) -> Dict:
        """Calculate the semivariogram by folds"""
        context_gamma = {}
        neighbors = [n for n in neighbors if n in self.train_data.index]
        neighbors_data = self.train_data.loc[neighbors]
        for fold, fold_data in neighbors_data.groupby(by=self.fold_col):
            similarity = self._calculate_similarity_matrix(fold_data, attribute)
            geo_weights = self._get_neighbors_weights(fold_data)
            gamma = self._calculate_gamma(similarity, geo_weights)
            context_gamma[fold] = {
                "gamma": gamma,
                "neighbors": fold_data.index.values.tolist(),
            }
        return context_gamma

    def _get_n_fold_neighbohood(self) -> int:
        """Get ne number of folds neighbors from the test set"""
        neighbors_idx = self._get_neighbors(self.test_data.index, self.adj_matrix)
        neighbors_idx = [n for n in neighbors_idx if n in self.data.index]
        return len(self.data.loc[neighbors_idx].groupby(self.fold_col))

    @staticmethod
    def _calculate_exponent(size_tree, count_n) -> np.float64:
        """Caclulate the decay exponent"""
        return np.log(1 * size_tree - count_n) / np.log(1 * size_tree)

    def _propagate_variance(self, attribute) -> List:
        """Calculate a buffer region"""
        # Initialize variables
        buffer = []  # containg the index of instaces buffered
        nodes_gamma = {fold: {} for fold in self.train_data[self.fold_col].unique()}
        # Start creating the buffer
        while len(buffer) < self.train_data.shape[0]:
            # Get the instance indexes from te test set + the indexes buffer
            growing_graph_idx = self.test_data.index.values.tolist() + buffer
            # Get the neighbor
            h_neighbors = self._get_neighbors(growing_graph_idx, self.adj_matrix)
            # Calculate the semivariogram for each fold in the neighborhood
            h_gamma = self._calculate_gamma_by_fold(h_neighbors, attribute)
            # Check for each fold in the neighborhood the semivariogram to decide
            # whether to add or not the instances into the buffer list
            for fold, h_fold_gamma in h_gamma.items():
                gamma = h_fold_gamma["gamma"]
                for node in h_fold_gamma["neighbors"]:
                    nodes_gamma[fold][node] = gamma
            buffer += h_neighbors
        return nodes_gamma
    
    def _calculate_buffer(self, nodes_propagated, type_attr):
        buffered_nodes = []
        sill = self.sill_target if type_attr == "target" else self.sill_reduced
        for fold, nodes in nodes_propagated.items():
            buffered_nodes += [key for key, value in nodes.items() 
                              if value <= sill[fold]]
        return buffered_nodes

    def run(self):
        """Generate graph-based spatial folds"""
        # Create folder folds
        start_time = time.time()
        name_folds = SRBUFFER if self.run_selection else RBUFFER
        self._init_fields()
        self._make_folders(["folds", name_folds])
        self.data[X_1DIM_COL] = self._calculate_train_pca()
        for fold_name, test_data in tqdm(
            self.data.groupby(by=self.fold_col), desc="Creating folds"
        ):
            # Cread fold folder
            self._mkdir(str(fold_name))
            # Initialize x , y and reduce
            self._split_data_test_train(test_data)
            # Calculate local sill
            self._initiate_buffers_sills()
            # Ensure indexes and columns compatibility
            self._convert_adj_matrix_index_types()
            # Calculate selection buffer
            if self.run_selection:
                nodes_prop  = nodes_prop = self._propagate_variance(X_1DIM_COL)
                selection_buffer = self._calculate_buffer(nodes_prop, "reduced")
                self.train_data = self.train_data.loc[selection_buffer]
            # The train data is used to calcualte the buffer. Thus, the size tree,
            # and the gamma calculation will be influenced by the selection buffer.
            # Calculate removing buffer
            nodes_prop = self._propagate_variance(self.target_col)
            removing_buffer = self._calculate_buffer(nodes_prop, "target")
            self.train_data.drop(index=removing_buffer, inplace=True)
            # Save buffered data indexes
            self._save_buffered_indexes(removing_buffer)
            # Save fold index relation table
            self._save_fold_by_index_training()
            # Clean data
            self._clean_data(cols_drop=[X_1DIM_COL, self.fold_col])
            # Save data
            self._save_data()
            # Update cur dir
            self.cur_dir = os.path.join(self._get_root_path(), "folds", name_folds)
        # Save execution time
        end_time = time.time()
        self._save_time(end_time, start_time)
        print(f"Execution time: {end_time-start_time} seconds")
