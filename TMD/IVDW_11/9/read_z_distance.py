from pymatgen.core import Structure

structure = Structure.from_file("POSCAR")
mo_positions = [site.coords for site in structure if site.species_string == "Mo"]
z_distance = abs(mo_positions[0][2] - mo_positions[-1][2])
print("Distance:", z_distance)

