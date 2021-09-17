"""Generate raw data for census"""
import os
from typing import Optional
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass, field
from src.classes.data import Data
from shapely.geometry import Point


@dataclass
class Merge(Data):
    """Represents the meshblock data in raw processing state.

    This object downloads meshblock data.

    Attributes
    ----------

    """

    census_filepath: str = None
    meshblock_filepath: str = None
    other_filepath: str = None
    type_merge: str = None
    save_filename: str = None
    left_id_col: str = None
    right_id_col: Optional[str] = None
    meshblock_id_col: Optional[str] = None
    meshblock_crs: Optional[int] = None
    __census: pd.DataFrame = field(default_factory=pd.DataFrame)
    __meshblock: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    __other_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    __merged_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _load_data(self):
        """Load initial dataset"""
        self.logger_info("Loading data to merge.")
        self.__census = pd.read_csv(self.census_filepath).infer_objects()
        self.__other_data = pd.read_csv(self.other_filepath).infer_objects()
        self.__meshblock = gpd.read_file(self.meshblock_filepath).infer_objects()

    def _normal_merge(self):
        """Normal merge"""
        self.logger_info("Merging data by normal join.")
        self.__merged_data = self.__census.merge(
            self.__other_data,
            left_on=self.left_id_col,
            right_on=self.right_id_col,
            how="inner",
            suffixes=("_DELETE", ""),
        )

    def _convert_other_data2gpd(self):
        """Convert other data to geopandas"""
        geometry = [
            Point((row["[GEO]_LONGITUDE"], row["[GEO]_LATITUDE"]))
            for _, row in self.__other_data.iterrows()
        ]
        self.__other_data = gpd.GeoDataFrame(
            self.__other_data, geometry=geometry, crs=self.meshblock_crs
        )

    def _spatial_merge(self):
        """Spatial merge"""
        self.logger_info("Merging data by spatial join.")
        self._convert_other_data2gpd()
        self.__other_data = gpd.sjoin(
            self.__other_data,
            self.__meshblock[[self.meshblock_id_col, "geometry"]],
            how="inner",
            op="within",
            lsuffix="_DELETE",
            rsuffix="",
        )
        self.__other_data.rename(columns={"geometry": "geometry_DELETE"}, inplace=True)
        self.right_id_col = self.meshblock_id_col
        self._normal_merge()

    def _merge_data(self):
        """Merge data"""
        map_merge_functions = {
            "normal": self._normal_merge,
            "spatial": self._spatial_merge,
        }
        map_merge_functions[self.type_merge]()

    def _check_nan(self):
        """Check for number of rows with Nan"""
        print(self.__merged_data.shape[0] - self.__merged_data.dropna().shape[0])

    def _clean_data(self):
        """Clean merged data"""
        self.logger_info("Cleaning merged.data")
        col_to_delete = [c for c in self.__merged_data if "DELETE" in c]
        self.__merged_data.drop(col_to_delete, axis=1, inplace=True)

    def _save_data(self):
        """Save merged data"""
        self.logger_info("Saving merged data.")
        self.__merged_data.to_csv(
            os.path.join(self.cur_dir, self.save_filename), index=False
        )

    def run(self) -> None:
        """Generate merged data"""
        self.init_logger_name(msg="Merging")
        self.logger_info("Generating Merged data.")
        self._make_folders(folders=[self.type_merge])
        self._load_data()
        self._merge_data()
        self._clean_data()
        self._save_data()
