from typing import List
from pandas import DataFrame, read_csv
from numpy import ndarray, isfinite
import os
import numpy as np
from deepmd import DeepMD, DeepMDModel
from ripser import Rips

class PrecomputeIVDW:
    def __init__(self, tmd_path:str = "TMD"):
        self.tmd_path = tmd_path
        self.total_dict = self.compute_dict()

    def compute_dict(self):
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

def read_deepmd_file(filepath: str | List[str]) -> DeepMDModel | List[DeepMDModel]:
    if isinstance(filepath, str):
        folder, mol_num = extract_deepmd_names(filepath).values()
        return DeepMD(folder, mol_num)
    
    models = []
    for file in filepath:
        folder, mol_num = extract_deepmd_names(file).values()
        models.append(DeepMD(folder, mol_num))
    return models

def extract_deepmd_names(file_path: str) -> tuple[str, str]:
    splits = file_path.split("/")
    tmd_ind = splits.index("TMD")
    splits = splits[tmd_ind+1:]
    folder, mol_num = splits[0], splits[1]
    return {"folder": folder, "file": mol_num}

def remove_infinity(data: List[ndarray]) -> List[ndarray]:
    return [d[isfinite(d).all(axis=1)] for d in data]

def construct_file_frame(file_dir:str) -> DataFrame:
    if "TMD" not in file_dir:
        raise ValueError("TMD must be in the file directory.")
        
    outpath = os.path.join(file_dir, "filepaths.csv")
    if os.path.exists(outpath):
        return read_csv(outpath)

    print("File doesn't exists. Creating now.")
    ivdw_folders = [(os.path.join(file_dir, folder), folder) for folder in os.listdir(file_dir) if folder != ".DS_Store"]
    mol_folders = []
    for ivdw_path, ivdw in ivdw_folders:
        for mol_folder in os.listdir(ivdw_path):
            if not mol_folder.endswith("_data"):
                final_path = os.path.join(ivdw_path, mol_folder)
                mol_folders.append(final_path) 
    frame = DataFrame(mol_folders, columns=["Filepath"])
    frame.to_csv(outpath, index=False)
    return frame