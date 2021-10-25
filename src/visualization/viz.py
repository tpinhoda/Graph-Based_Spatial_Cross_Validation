"""Training data process"""
import os
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm
from src.data import Data
import src.utils as utils


@dataclass
class Visualization(Data):
    pass
