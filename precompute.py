from typing import Dict
import pandas as pd
import numpy as np
from ripser import Rips
from tqdm import tqdm
import os
from utils import remove_infinity
from deepmd import DeepMD
from dataclasses import dataclass

@dataclass
class Precompute:
    box: pd.DataFrame
    energy: pd.DataFrame
    force: pd.DataFrame
    type: pd.DataFrame
    virial: pd.DataFrame
    rips: pd.DataFrame
    homology: int

class PrecomputeIVDW:
    def __init__(self, hm:int, tmd_path:str = "TMD"):
        self.tmd_path = tmd_path
        self.total_dict = self.compute_dict()
        self.precompute = self.collect_frames(hm=hm)

    def compute_dict(self) -> Dict[str, Dict]:
        filename_dicts = {
            "box": "TMD/dicts/box_dict.npy",
            "energy": "TMD/dicts/energy_dict.npy",
            "force": "TMD/dicts/force_dict.npy",
            "rips": "TMD/dicts/rips_dict.npy",
            "type": "TMD/dicts/type_dict.npy",
            "virial": "TMD/dicts/virial_dict.npy"
        }

        # check if all data has been precomputed
        final_dict = {}
        if all(os.path.exists(fd) for fd in list(filename_dicts.values())):
            for name, fd in filename_dicts.items():
                final_dict[name] = np.load(fd, allow_pickle=True)
            return final_dict
        
        print("Data has not been precomputed. This may take ~30mins.")
        rips = Rips(maxdim=2, verbose=False)

        # precompute tmd data
        filepath = "TMD"
        folders = []
        for folder in os.listdir(filepath):
            if folder not in [".DS_Store", "filepaths.csv"]:
                folder_path = os.path.join(filepath, folder)
                folders.append(folder_path)
        folders = list(set(folders))

        box_dict = {}
        energy_dict = {}
        force_dict = {}
        type_dict = {}
        virial_dict = {}
        rips_complexes = {}
        for folder in folders:
            for mols in os.listdir(folder):
                if not mols.endswith("_data"):
                    path = os.path.join(folder, mols)
                    print(path)
                    curr_folder = folder.lstrip("TMD/")
                    model = DeepMD(curr_folder, mols).model
                    box_dict[path] = model.box
                    energy_dict[path] = model.energy
                    force_dict[path] = model.force
                    type_dict[path] = model.type
                    virial_dict[path] = model.virial
                    rips_complexes[path] = {}
                    for i, coord in enumerate(model.coord):
                        rips_complexes[path][i] = {}
                        rips_cell = rips.fit_transform(coord)
                        rips_cell = remove_infinity(rips_cell)
                        for j, cells in enumerate(rips_cell):
                            rips_complexes[path][i][j] = cells

        # save the files
        tmd_dict = {
            "energy": energy_dict,
            "box": box_dict,
            "force": force_dict,
            "type": type_dict,
            "virial": virial_dict,
            "rips": rips_complexes
        }
        np.save("TMD/dicts/energy_dict.npy", energy_dict, allow_pickle=True)
        np.save("TMD/dicts/box_dict.npy", box_dict, allow_pickle=True)
        np.save("TMD/dicts/force_dict.npy", force_dict, allow_pickle=True)
        np.save("TMD/dicts/type_dict.npy", type_dict, allow_pickle=True)
        np.save("TMD/dicts/virial_dict.npy", virial_dict, allow_pickle=True)
        np.save("TMD/dicts/rips_dict.npy", rips_complexes, allow_pickle=True)
        return tmd_dict

    def compute_frame(self, name:str, save_path:str = "TMD/frames") -> pd.DataFrame:
        filename = f"{name}_frame.parquet"
        out_path = os.path.join(save_path, filename)
        if os.path.exists(out_path):
            print(f"{filename} exists. Loading now.")
            return pd.read_parquet(out_path, engine='pyarrow')

        print(f"{filename} doesn't exist. Creating now.")
        data = list(self.total_dict[name][()].items())
        num_files = len(data)
        frames = []
        for n in range(num_files):
            curr = data[n]
            curr_frame = pd.DataFrame(curr[1].tolist())
            curr_frame['path'] = curr[0]
            path_column = curr_frame['path'].apply(lambda x: x.split("/")[1:])
            path_frame = pd.DataFrame(path_column.tolist(), columns = ['Folder', 'mol_num'])
            new_frame = pd.concat([path_frame, curr_frame], axis = 1).reset_index()
            new_frame = new_frame.rename(columns = {"index": "transition"})
            new_frame = new_frame.melt(id_vars=["path", "Folder", "mol_num", 'transition'], var_name=f"{name}_id", ignore_index=True)
            frames.append(new_frame)

        total_frame = pd.concat(frames, axis = 0).sort_values(['Folder', 'mol_num', 'transition', f'{name}_id']).reset_index(drop = True)

        # save data
        print(f"Saving {filename}.")
        total_frame.to_parquet(out_path, compression='gzip', index=False)
        return total_frame
    
    def compute_rips_frame(self, hm:int, save_path:str = "TMD/frames") -> pd.DataFrame:
        filename = f"rips_{hm}_frame.parquet"
        out_path = os.path.join(save_path, filename)
        if os.path.exists(out_path):
            print(f"{filename} exists. Loading now.\n")
            return pd.read_parquet(out_path, engine='pyarrow')

        print(f"{filename} doesn't exist. Creating now.")
        data = list(self.total_dict['rips'][()].items())
        num_files = len(data)
        total_frames = []
        for n in tqdm(range(num_files)):
            path, cell_data = data[n]
            hom_data_frames = []
            for pers_idx, hom_data in cell_data.items():
                curr_hom_data = pd.DataFrame(hom_data[hm].tolist(), columns=['birth', 'death']).reset_index()
                curr_hom_data = curr_hom_data.rename(columns={"index": "pers_id"})
                curr_hom_data['transition'] = pers_idx
                hom_data_frames.append(curr_hom_data)    
            concat_hom_data = pd.concat(hom_data_frames, axis = 0, ignore_index=True)
            concat_hom_data['path'] = path
            path_column = concat_hom_data['path'].apply(lambda x: x.split("/")[1:])
            path_frame = pd.DataFrame(path_column.tolist(), columns = ['Folder', 'mol_num'])
            concat_hom_data = pd.concat([path_frame, concat_hom_data], axis = 1)
            total_frames.append(concat_hom_data)
            
        total_frames_concat = pd.concat(total_frames, axis = 0, ignore_index=True)\
            .sort_values(['Folder', 'mol_num', "transition", "pers_id"])\
            .reset_index(drop = True)
        print(f"Saving {filename}.\n")
        total_frames_concat.to_parquet(out_path, compression="gzip", index=False)
        return total_frames_concat
    
    def collect_frames(self, hm:int) -> Precompute:
        print(f"Collecting Frames for {hm}-Homology...\n")
        print("Collecting residual frames...")
        energy_frame = self.compute_frame('energy')
        box_frame = self.compute_frame('box')
        force_frame = self.compute_frame('force')
        type_frame = self.compute_frame('type')
        virial_frame = self.compute_frame('virial')
        print()
        print(f"Creating Rips Frame for {hm}-Homology...")
        rips_frame = self.compute_rips_frame(hm=hm)
        return Precompute(
            box=box_frame,
            energy=energy_frame,
            force=force_frame,
            type=type_frame,
            virial=virial_frame,
            rips=rips_frame,
            homology=hm
        )
