import os 
from dataclasses import dataclass
import numpy as np
from numpy import ndarray

@dataclass
class DeepMDModel:
    box: ndarray
    coord: ndarray
    energy: ndarray
    force: ndarray
    type: ndarray
    virial: ndarray

class DeepMD:
    def __init__(self, folder: str, mol_num:str) -> None:
        self.folder = self._validate_folder(folder)
        self.mol_num = self._validate_mol_num(mol_num)
        self.base_path = os.path.join("TMD", self.folder, self.mol_num, "data_deepmd")
        self.model = self._collect_data()

    def _validate_folder(self, folder: str) -> str:
        folders = os.listdir("TMD")
        if folder not in folders:
            raise ValueError(
                f"{folder} is not an accepted folder."
                "Look in the TMD directory."
            )
        return folder
    
    def _validate_mol_num(self, mol_num: str) -> str:
        folder_path = os.path.join("TMD", self.folder)
        mol_nums = os.listdir(folder_path)
        if mol_num not in mol_nums:
            raise ValueError(
                f"{mol_num} is not an accepted Mol num."
                f"Look in the {self.folder} directory."
            )
        return mol_num
    
    def _convert_type(self, types: ndarray, type_map: ndarray) -> ndarray:
        # build dictionary
        type_map_values = type_map.flatten().tolist()
        types_dict = {i: tm for i, tm in enumerate(type_map_values)}

        types_flatten = types.flatten().tolist()
        f = lambda x: types_dict[x]
        new_types = np.array(list(map(f, types_flatten)))
        return new_types

    def _collect_data(self) -> None:
        datadict = {}
        filenames = os.listdir(self.base_path)
        for filename in filenames:
            if not filename.endswith(".raw"):
                continue
            filepath = os.path.join(self.base_path, filename)
            # read in the file
            with open(filepath, 'r+', encoding='utf-8') as file:
                lines = file.readlines()
                peek = lines[0].split()
                #if array is (x, y)
                if len(peek) != 1:
                    data = np.array([np.array(line.split(" "), dtype=float) for line in lines])

                # if array is (x, 1)
                else:
                    lines = [line.rstrip("\n") for line in lines]
                    # if array of letters an non numbers
                    if lines[0].isalpha():
                        data = np.array(lines)
                    else:
                        data = np.array(lines, dtype=float)
                    data = np.reshape(data, (-1, 1))
                
                property_name = filename.rstrip(".raw")

                if property_name in ['coord', 'force']:
                    new_shape = (-1, data.shape[1]//3, 3)
                    data = data.reshape(new_shape)
                datadict[property_name] = data

        # convert the types to symbol
        if "type" in datadict:
            datadict['type'] = self._convert_type(datadict['type'], datadict['type_map'])
        
        # remove the type map as it is not needed anymore
        del datadict['type_map']
        return DeepMDModel(**datadict)


