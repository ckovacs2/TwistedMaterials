from collections import defaultdict
from typing import List
import torch
from torch.utils.data import Dataset
from numpy import ndarray
import numpy as np
from dataclasses import dataclass
from numpy import ndarray
from utils import construct_file_frame
from persistence import RipsComplexDeepMD
from embeddings import PBoW

__all__ = [
    "PBowDataset"
]


@dataclass
class PersistenceSample:
    """Dataclass to hold information of Persistence Sample for Machine Learning."""

    data: ndarray
    force: ndarray
    energy: ndarray

class PBowDataset(Dataset):
    def __init__(
            self,
            data_dir: str = 'TMD', 
            hm_class: int = 2,
            nclusters: int = 5
        ):
        self.fileframe = construct_file_frame(data_dir)
        self.hm_class = hm_class
        self.nclusters = nclusters
    def __len__(self) -> int:
        return len(self.fileframe)
    
    def __str__(self) -> str:
        return f"PBoW Dataset for the {self.hm_class}-Homology"
    
    def _get_rips_complex(self, idx: int) -> tuple[RipsComplexDeepMD, ndarray]:
        filepath = self.fileframe.loc[idx, "Filepath"]
        rips_complex = RipsComplexDeepMD(filepath)
        force, energy = rips_complex.pt_cloud.force, rips_complex.pt_cloud.energy
        return rips_complex, force, energy
    
    def _get_features(self, rips_complex: RipsComplexDeepMD) -> ndarray:
        instance = PBoW(
            rips_complex=rips_complex, 
            nclusters=self.nclusters
        )
        features = instance.features[:, self.hm_class, :]

        return features
    
    def _get_sample(self, idx: int | List[int]) -> PersistenceSample:
        if not isinstance(idx, list):
            rips_complex, force, energy = self._get_rips_complex(idx=idx)
            feature = self._get_features(rips_complex)
            sample = {
                "data": feature,
                "force": force,
                "energy": energy
            }
            return PersistenceSample(**sample)

        sample = defaultdict(list)
        for ind in idx:
            rips_complex, force, energy = self._get_rips_complex(idx=ind)
            feature = self._get_features(rips_complex)
            sample["data"].append(feature)
            sample['force'].append(force)
            sample['energy'].append(energy)

        sample["data"] = np.array(sample["data"])
        sample['force'] = np.array(sample['force'])
        sample['energy'] = np.array(sample['energy'])
        return PersistenceSample(**sample)

    def __getitem__(self, idx) -> PersistenceSample:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._get_sample(idx)
