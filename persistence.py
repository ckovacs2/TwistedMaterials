from typing import Any, List
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ripser import Rips
from persim import plot_diagrams
from numpy import ndarray
from deepmd import DeepMDModel
from utils import extract_deepmd_names, read_deepmd_file, remove_infinity
__all__ = ["RipsComplexDeepMD"]

class RipsComplexDeepMD:
    def __init__(self, filepath: str) -> None:
        self.pt_cloud = self._get_pt_cloud(filepath).model
        self.diag = self._produce_diag(self.pt_cloud.coord)
        self.meta_ = self._assign_meta(filepath)

    def _assign_meta(self, filepath: str) -> dict[str, Any]:
        meta = extract_deepmd_names(filepath)
        meta["cell"] = self.pt_cloud.box
        meta["super_cell"] = False
        return meta

    def _get_pt_cloud(self, path: str) -> DeepMDModel:
        if "TMD" not in path:
            raise ValueError(f"{path} is not an accepted path."
                             "Must be from the TMD folder.")
        return read_deepmd_file(path)            

    def _produce_diag(self, coords: ndarray) -> List[List[ndarray]]:
        """Generate the persistence diagrams from the given point cloud or distance data.

        Args:
            data (np.ndarray): Point cloud or distance matrix.
            distance (bool, optional): Tells method if data is distance matrix. Defaults to False.

        Returns:
            list[np.ndarray]: Persistence diagrams of the 0, 1, and 2-Homology.
        """
        rips = Rips(maxdim=2, verbose=False)
        rips_data = []
        for i in range(coords.shape[0]):
            rips_cell = rips.fit_transform(coords[i, ...])
            rips_cell = remove_infinity(rips_cell)
            rips_data.append(rips_cell)
        return rips_data

    def plot(self, num_cell:int = None) -> None:
        """Plotting persistence diagram of the rips complex"""
        mpl.rcParams.update(mpl.rcParamsDefault)
        folder, filename = self.meta_["folder"], self.meta_["file"]
        title = "/".join([folder, filename])
        if num_cell is None:
            num_cell = 0
        plot_diagrams(self.diag[num_cell], title=title + f" | Unit cell #{num_cell}")
        plt.show()

        return None

