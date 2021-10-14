"""Generate optmistic spatial folds"""
import os
import time
from dataclasses import dataclass
from tqdm import tqdm
from src.scv.scv import SCV


@dataclass
class OPTMISTIC(SCV):
    """Represents the Optmistic Spatial Cross-Validation, without considering
    the removing buffer.

     Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
        root_path : str
            Root path
    """

    def create_folds(
        self, run_selection=None, name_folds="optmistic", kappa=None
    ) -> None:
        """Generate merged data"""
        # Create folder folds
        start_time = time.time()
        self._make_folders([name_folds])
        for fold_name, test_data in tqdm(self.data.groupby(by=self.fold_col)):
            # Cread fold folder
            self._mkdir(str(fold_name))
            # Initialize x , y and reduce
            self._split_data_test_train(test_data)
            # Save buffered data indexes
            self._save_buffered_indexes(removing_buffer=[])
            # Clean data
            self._clean_data(cols_drop=[self.fold_col])
            # Save data
            self._save_data()
            # Update cur dir
            self.cur_dir = os.path.join(self._get_root_path(), name_folds)
        end_time = time.time()
        print(f"Execution time: {end_time-start_time} seconds")