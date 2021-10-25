"""Pipeline to analyse electoral data"""
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional
import inspect
import pandas as pd
from src.scv.optimistic import Optimistic
from src.scv.gbscv import GraphBasedSCV
from src.scv.ultra_coservative import UltraConservative
from src.feature_selection.fs import FeatureSelection
from src.model.train import Train
from src.model.predict import Predict
from src.model.evaluate import Evaluate
from src.visualization.viz import Visualization

PIPELINE_MAP: Final = {
    "scv": {
        "UltraConservative": UltraConservative,
        "RBuffer": GraphBasedSCV,
        "SRBuffer": GraphBasedSCV,
        "Optimistic": Optimistic,
    },
    "fs": FeatureSelection,
    "train": Train,
    "predict": Predict,
    "evaluate": Evaluate,
    "visualization": Visualization,
}


@dataclass
class Pipeline:
    """Represents a pipeline to evaluate data.

    This object evaluate spatial data.

    Attributes
    ----------
    data_name: str
        Describes the type of data [location or results]
    params: Dict[str, str]
        Dictionary of parameters
    switchers: Dict[str, int]
        Dictionary of switchers to generate the pipeline
    """

    root_path: str = None
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_col: str = None
    fold_col: str = None
    target_col: str = None
    scv_method: str = None
    run_selection: Optional[bool] = None
    kappa: Optional[float] = None
    fs_method: str = None
    ml_method: str = None
    paper: bool = False
    switchers: Dict[str, str] = field(default_factory=dict)
    __pipeline: List[str] = field(default_factory=list)

    @staticmethod
    def _get_class_attributes(class_process):
        """Returns the attributes required to instanciate a class"""
        attributes = inspect.getmembers(
            class_process, lambda a: not inspect.isroutine(a)
        )
        attributes = [
            a[0]
            for a in attributes
            if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        return [attr for attr in attributes if not attr.startswith("_")]

    def _get_parameter_value(self, attributes):
        """Get parameter values"""
        params = {
            "root_path": self.root_path,
            "data": self.data,
            "adj_matrix": self.adj_matrix,
            "index_col": self.index_col,
            "fold_col": self.fold_col,
            "target_col": self.target_col,
            "scv_method": self.scv_method,
            "run_selection": self.run_selection,
            "kappa": self.kappa,
            "fs_method": self.fs_method,
            "ml_method": self.ml_method,
        }
        return {attr: params.get(attr) for attr in attributes}

    def _generate_parameters(self, process):
        """Generate parameters dict"""
        attributes = self._get_class_attributes(process)
        return self._get_parameter_value(attributes)

    def _get_init_function(self, process):
        """Return the initialization fucntion"""
        return PIPELINE_MAP[process]

    def _init_class(self, process):
        """Initialize a generic class"""
        if process == "scv":
            data_class = self._get_init_function("scv")[self.scv_method]
        else:
            data_class = self._get_init_function(process)
        parameters = self._generate_parameters(data_class())
        return data_class(**parameters)

    def get_pipeline_order(self):
        """Return pipeline order"""
        return [process for process in self.switchers if self.switchers[process]]

    def map_pipeline_process(self, process):
        """Map the process initialization functions"""
        processes = {
            "scv": self._init_class,
            "fs": self._init_class,
            "train": self._init_class,
            "predict": self._init_class,
            "evaluate": self._init_class,
            "vizualization": self._init_class,
        }
        return processes[process](process)

    def generate_pipeline(self):
        """Generate pipeline to process data"""
        pipeline_order = self.get_pipeline_order()
        for process in pipeline_order:
            self.__pipeline.append(self.map_pipeline_process(process))

    def run(self):
        """Run pipeline"""
        self.generate_pipeline()
        for process in self.__pipeline:
            process.run()