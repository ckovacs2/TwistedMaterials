# TwistedMaterials

This repository uses embedding methods to understand transitions of 2-dimensional materials.

## Data
The `TMD` folder has subfolders labeled `IVDW_X`. The `_X` at the end of each folder corresponds to a different sampling method described [here](https://www.vasp.at/wiki/index.php/IVDW). In this case, we have 3 different sampling methods:
- `IVDW_4` corresponds to the [dDsC dispersion correction](https://www.vasp.at/wiki/index.php/DDsC_dispersion_correction) method;
- `IVDW_10` corresponds to the [DFT-D2](https://www.vasp.at/wiki/index.php/DFT-D2) method of Grimme;
- `IVDW_11` corresponds to the [DFT-D3](https://www.vasp.at/wiki/index.php/DFT-D3) method of Grimme with zero-damping function.


Each folder contains a molecule listed 0-31 containing different molecular structures. Within each molecular folder contains the following measurements per the given sampling technique:
- `box.raw`: The lattice constants, vectors of the three directions of the crystal's smallest unit, for each structure, each line unit is in $\mathring{A}$ (Angstrom). Detailed description can be found [here](https://metal.elte.hu/~groma/Anyagtudomany/kittel.pdf). 
- `cood.raw`: The coordinate of all atoms for each structure where each line unit is in $\mathring{A}$ (Angstrom)
- `energy.raw`: The energy of each structure where each line unit is in  $eV$ (electron-Volts)
- `force.raw`: The force of all atoms for each structure where each line unit is in  $\frac{eV}{\mathring{A}}$ (electron-Volts per Angstrom)
- `type.raw`: The type number of each atoms
- `type_map.raw`: The atom symbol of each type number

**For example**, the path `TMD\IVDW_4\0\energy.raw` collections all of the energy of each structure in $eV$ for the molecule $\text{Te}_{36}\text{Mo}_{18}$. In this case, 2000 observations were made for this molecule.
