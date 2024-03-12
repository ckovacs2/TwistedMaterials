from typing import Literal
from torch.utils.data import Dataset
from pandas import DataFrame
from typing import Dict
import torch
from tqdm import tqdm
import pandas as pd
import os
from numpy import ndarray
from embeddings import PBoW
from precompute import PrecomputeIVDW

__all__ = [
    "PBowDataset"
]


class PBowDataset(Dataset):
    def __init__(
            self,
            dataset_type: Literal['train','val', 'test'],
            data_dir: str = 'TMD', 
            hm_class: int = Literal[1, 2],
            nclusters: int = Literal[3, 5, 10],
        ):
        self.embeddings_path = f"model_data/{dataset_type}-{hm_class}-{nclusters}_pbow_embeddings.parquet"
        self.hm_class = hm_class
        self.nclusters = nclusters
        self.precompute = PrecomputeIVDW(hm=self.hm_class, tmd_path=data_dir).precompute
        self.fileframe, self.filepaths = self._get_file_frame(datatype=dataset_type)
        self.pbow = self._get_pbow_instance(self.precompute.rips)
        self.force, self.energy = self._get_labels()
        self.features = self._get_features()

    def __len__(self) -> int:
        return len(self.features)
    
    def __str__(self) -> str:
        return f"PBoW Dataset for the {self.hm_class}-Homology"
    
    def _validate_path(self, path: str) -> bool:
        if os.path.exists(path):
            return True
        return False
    
    def _get_file_frame(self, datatype: str = Literal['train', 'val', 'test']) -> DataFrame:
        splits = self.precompute.splits
        print(f"Getting {datatype} data...\n")
        fileframe = splits[splits['data_type'] == datatype].reset_index(drop = True)
        return fileframe, fileframe['path'].values.tolist()
    
    def _get_pbow_instance(self, rips_data: DataFrame) -> PBoW:
        self.curr_rips_data = rips_data[rips_data['path'].isin(self.filepaths)]
        return PBoW(self.curr_rips_data.loc[:, ['birth', 'death']].values, nclusters=self.nclusters)

    def _get_labels(self) -> tuple[ndarray, ndarray]:
        force_frame = self.precompute.force[self.precompute.force['path'].isin(self.filepaths)].reset_index(drop = True)
        energy_frame = self.precompute.energy[self.precompute.force['path'].isin(self.filepaths)].reset_index(drop = True)
        force = force_frame.iloc[:, 4:].values
        energy = energy_frame['energy'].values
        return force, energy
    
    def _get_rips_complex(self) -> ndarray:
        rips_frame = self.precompute.rips[self.precompute.rips['path'].isin(self.filepaths)].reset_index(drop = True)
        rips = rips_frame[['path', 'transition', 'birth', 'death']]
        rips_groupings = {
            k: rips_frame.loc[frame, ['birth', 'death']].values for k, frame in tqdm(rips.groupby(['path', 'transition']).groups.items())
        }
        return rips_groupings
    
    def _get_features(self):

        if not self._validate_path(self.embeddings_path):
            print(f"{self.embeddings_path} doesn't exist. Collecting and creating now...")
            rips = self._get_rips_complex()
            feature = self.pbow.features(rips)
            feature_frame = pd.DataFrame(data = feature, columns = range(self.nclusters))
            feature_frame.to_parquet(self.embeddings_path, compression='gzip', index = False)
            return feature_frame

        feature_frame = pd.read_parquet(self.embeddings_path, engine='pyarrow')
        return feature_frame
    
    def __getitem__(self, idx) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {
            "data": self.features.iloc[idx, :].values,
            "force": self.force[idx, :],
            "energy": self.energy[idx]
        }
        return sample
