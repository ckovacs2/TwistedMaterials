from typing import List
from pandas import DataFrame, read_csv
from numpy import ndarray, isfinite
import os
from deepmd import DeepMD, DeepMDModel



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