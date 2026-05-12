from pathlib import Path

energy_output_file = Path("./all_energy_files.txt")
force_output_file = Path("./all_force_files.txt")

energy_files = Path(".").rglob("*_energy.xyz")

with energy_output_file.open("w") as eo, force_output_file.open("w") as fo:
    for file in energy_files:
        eo.write(str(file.parent) + "\n")  
        eo.write(file.read_text())         
        eo.write("\n\n")                   

        ffile = list(file.parent.rglob("*_force.xyz"))[0]
        fo.write(str(file.parent) + "\n") 
        fo.write(ffile.read_text())        
        fo.write("\n")                   

        print(f"finishing {str(file)}")
