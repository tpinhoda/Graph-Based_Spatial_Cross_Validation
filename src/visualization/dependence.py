"""Dependence visualization class"""
import os
from dataclasses import dataclass, field
from typing import Dict, List
from operator import itemgetter
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pylab as plt
from tqdm import tqdm
from src.data import Data
from src import utils


@dataclass
class VizDependence(Data):
    """Generates plots to visualize dependence between test and training folds.

    This object generates plots to visualize depdencence between test and training folds.

    Attributes
    ----------
    root_path : str
        Root path
    index_col : str
        The data index column name
    """

    cv_methods: List = field(default_factory=list)
    index_col: str = None
    fold_col: str = None
    target_col: str = None
    prob: float = None
    fold_list: List = field(default_factory=list)
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper: bool = False
    _train: pd.DataFrame = field(default_factory=pd.DataFrame)
    _test: pd.DataFrame = field(default_factory=pd.DataFrame)
    _fold_idx: pd.DataFrame = field(default_factory=pd.DataFrame)
    _split_data: Dict = field(default_factory=dict)
    _cv_methods_path: List = field(default_factory=list)
    _boundary: pd.DataFrame = field(default_factory=pd.DataFrame)
    _dependence: pd.DataFrame = field(default_factory=pd.DataFrame)
    _tosee: Dict = field(default_factory=dict)

    def _init_methods_path(self):
        """Initialize spatial cv folder paths"""
        self._cv_methods_path = [
            os.path.join(self.root_path, "folds", method) for method in self.cv_methods
        ]

    def _read_train_data(self, folds_path, fold, paper):
        """Read train data"""
        self._train = self._read_data(folds_path, fold, "train.ftr")
        if paper:
            cols = [c for c in self._train.columns if "CENSUS" in c] + [self.target_col]
            self._train = self._train[cols]

    def _read_test_data(self, folds_path, fold, paper):
        """Read test data"""
        self._test = self._read_data(folds_path, fold, "test.ftr")
        if paper:
            cols = [c for c in self._test.columns if "CENSUS" in c] + [self.target_col]
            self._test = self._test[cols]

    def _read_data(self, folds_path, fold, filename):
        """Read generic data"""
        file_path = os.path.join(folds_path, fold, filename)
        data = pd.read_feather(file_path)
        data.set_index(self.index_col, inplace=True)
        return data

    def _read_split_data(self, folds_path, fold):
        """Read split data json file"""
        json_path = os.path.join(folds_path, fold, "split_data.json")
        self._split_data = utils.load_json(json_path)

    def _read_fold_idx_table(self, folds_path, fold):
        """Read the fold_by_idx data"""
        self._fold_idx = pd.read_csv(
            os.path.join(folds_path, fold, "fold_by_idx.csv"), index_col=self.index_col
        )

    def _initialize_data(self, folds_path, fold, paper):
        """Load the data"""
        self._read_train_data(folds_path, fold, paper)
        self._read_test_data(folds_path, fold, paper)
        self._read_split_data(folds_path, fold)
        self._read_fold_idx_table(folds_path, fold)

    def _initialize_dependence_df(self):
        """Initialize dependence dataframe"""
        self._dependence = pd.DataFrame(columns=self.cv_methods + ["FOLDS"])
        self._dependence["FOLDS"] = self.fold_list
        self._dependence.set_index("FOLDS", inplace=True)
        self._dependence.fillna(0, inplace=True)

    def _convert_adj_matrix_index_types(self):
        """Convert adjacenty matrixy index and columns types to the same as in the data"""
        self.adj_matrix.index = self.adj_matrix.index.astype(self._train.index.dtype)
        self.adj_matrix.columns = self.adj_matrix.columns.astype(
            self._train.index.dtype
        )

    @staticmethod
    def _get_neighbors(indexes, adj_matrix):
        """Return the 1-degree neighborhood from a given sub-graph formed by indexes"""
        area_matrix = adj_matrix.loc[indexes]
        neighbors = area_matrix.sum(axis=0) > 0
        neighbors = neighbors[neighbors].index
        neighbors = [n for n in neighbors if n not in indexes]
        return neighbors

    def _get_boundary(self):
        """Returns spatial objects in the boundary of removing + test data"""
        indexes = self._split_data["removing_buffer"] + self._split_data["test"]
        neighbors = self._get_neighbors(indexes, self.adj_matrix)
        neighbors = [n for n in neighbors if n not in self._split_data["discarded"]]
        neighbors = [n for n in neighbors if n in self._fold_idx.index]
        self._boundary = self._fold_idx.loc[neighbors].copy()

    def _get_n_nearest_folds(self, n_folds):
        """Returns the nearest n folds in the boundary"""
        dist_fold = {}
        mean_test = self._test.drop(columns=[self.target_col]).mean(axis=0)
        boundary_folds = list(self._boundary.groupby(self.fold_col).groups.keys())
        for fold in boundary_folds:
            indexes = self._fold_idx[self._fold_idx[self.fold_col] == fold].index
            mean_fold_train = (
                self._train.loc[indexes].drop(columns=[self.target_col]).mean(axis=0)
            )
            dist_fold[fold] = ((mean_test - mean_fold_train) ** 2).sum()
        return dict(sorted(dist_fold.items(), key=itemgetter(1))[:n_folds])

    @staticmethod
    def _get_observations(target):
        """Count observations of win and lost"""
        unique, counts = np.unique(target, return_counts=True)
        dict_obs = dict(zip(unique, counts))
        if not dict_obs.get(0):
            dict_obs[0] = 1
        elif not dict_obs.get(1):
            dict_obs[1] = 1
        return [dict_obs[0], dict_obs[1]]

    def _calculate_dependence(self, n_folds, fold, method):
        """Calculate dependence dataframe"""
        test_target = np.where(self._test[self.target_col] > 50, 1, 0)
        test_obs = self._get_observations(test_target)
        pvalores = []
        for n_fold in n_folds:
            fold_idx = self._fold_idx[self._fold_idx[self.fold_col] == n_fold].index
            fold_target = np.where(
                self._train.loc[fold_idx, self.target_col] > 50, 1, 0
            )
            fold_obs = self._get_observations(fold_target)
            obs = np.array([fold_obs, test_obs])
            _, p_value, _, _ = chi2_contingency(obs)
            pvalores.append(p_value)
            alpha = 1.0 - self.prob
            if p_value >= alpha and len(fold_idx) > 0:
                self._dependence.loc[int(fold), method] += 1
        if fold == "43":
            self._tosee[method] = pvalores
            

    def _generate_dependence_plot(self):
        """Generates dependece heatmap"""
        sns.set(font_scale=2.4)
        fig, ax_fig = pyplot.subplots(figsize=(22, 19))
        cmap = sns.diverging_palette(120, 0, 100, 50, as_cmap=True)
        sns.heatmap(
            self._dependence.T,
            ax=ax_fig,
            cmap=cmap,
            vmin=0,
            vmax=3,
            square=True,
            linewidth=0.3,
            cbar_kws={"shrink": 0.16},
        )
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        file_path = os.path.join(self.cur_dir, "dependence_heatmap.pdf")
        fig.savefig(file_path, dpi=300, bbox_inches="tight")

    def run(self):
        """Runs de visualization process"""
        self._init_methods_path()
        self._make_folders(["comparison"])
        self._initialize_dependence_df()
        for method_path, method in tqdm(
            zip(self._cv_methods_path, self.cv_methods), total=len(self.cv_methods)
        ):
            list_folds = self._get_folders_in_dir(method_path)
            for fold in list_folds:
                self._initialize_data(method_path, fold, self.paper)
                self._convert_adj_matrix_index_types()
                self._get_boundary()
                n_folds = self._get_n_nearest_folds(n_folds=3)
                self._calculate_dependence(n_folds.keys(), fold, method)
        self._dependence.to_csv(os.path.join(self.cur_dir, "dependence.csv"))
        self._generate_dependence_plot()
