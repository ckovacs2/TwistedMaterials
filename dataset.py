from typing import List, Literal
import torch
from torch.utils.data import Dataset
from pandas import DataFrame
from dataclasses import dataclass
from numpy import ndarray
from embeddings import PBoW
from precompute import PrecomputeIVDW, Precompute

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
            dataset_type: Literal['train','val', 'test'],
            data_dir: str = 'TMD', 
            hm_class: int = 2,
            nclusters: int = 5
        ):
        self.precompute = PrecomputeIVDW(hm=hm_class, tmd_path=data_dir).precompute
        self.fileframe = self._get_file_frame(self.precompute, datatype=dataset_type)
        self.hm_class = hm_class
        self.nclusters = nclusters
        self.pbow = self._get_pbow_instance(self.precompute.rips)

    def __len__(self) -> int:
        return len(self.fileframe)
    
    def __str__(self) -> str:
        return f"PBoW Dataset for the {self.hm_class}-Homology"
    
    def _get_file_frame(self, precompute: Precompute, datatype: str = Literal['train', 'val', 'test']) -> DataFrame:
        splits = precompute.splits
        return splits[splits['data_type'] == datatype].reset_index(drop = True)
    
    def _get_pbow_instance(self, rips_data: DataFrame) -> PBoW:
        paths = self.fileframe.path.values.tolist()
        self.curr_rips_data = rips_data[rips_data['path'].isin(paths)]
        return PBoW(self.curr_rips_data.loc[:, ['birth', 'death']].values)

    def _get_labels(self, idx: int | list[int]) -> tuple[ndarray, ndarray]:
        filepaths = self.fileframe.loc[idx, "path"].values
        if isinstance(idx, int):
            force_frame = self.precompute.force[self.precompute.force['path'] == filepaths].reset_index(drop = True)
            energy_frame = self.precompute.energy.loc[self.precompute.force['path'] == filepaths].reset_index(drop = True)
        else:
            force_frame = self.precompute.force[self.precompute.force['path'].isin(filepaths)].reset_index(drop = True)
            energy_frame = self.precompute.energy[self.precompute.force['path'].isin(filepaths)].reset_index(drop = True)
        force = force_frame.iloc[idx, 4:].values
        energy = energy_frame.loc[idx, 'energy'].values
        return force, energy
    
    def _get_rips_complex(self, idx: int | list[int]) -> ndarray:
        filepaths = self.fileframe.loc[idx, "path"].values
        if isinstance(idx, int):
            rips_frame = self.precompute.rips[self.precompute.rips['path'] == filepaths].reset_index(drop = True)
        else:
            rips_frame = self.precompute.rips[self.precompute.rips['path'].isin(filepaths)].reset_index(drop = True)
        rips = rips_frame.loc[idx, ['birth', 'death']].values.tolist()
        return rips
    
    def _get_features(self, diags) -> ndarray:
        features = self.pbow.features(diags)
        return features
    
    def _get_sample(self, idx: int | List[int]) -> PersistenceSample:
        force, energy = self._get_labels(idx=idx)
        rips = self._get_rips_complex(idx=idx)
        feature = self._get_features(rips)
        sample = {
            "data": feature,
            "force": force,
            "energy": energy
        }
        return PersistenceSample(**sample)
    
    def __getitem__(self, idx) -> PersistenceSample:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._get_sample(idx)
