#coding=utf-8
import json
import re, sys
import math
import fnmatch

def parse_range_list(s):
    result = []
    for part in s.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result

def replace_lammps_params(input_file, output_file, params_file):
    # Atomic type mapping
    elem_mapping = {
        "Si": 1,
        "O" : 2,
        "V" : 3,
        "C" : 4,
        "N" : 5,
        "Na": 6
    }

    # Load the data file
    with open(params_file) as f:
        params = json.load(f)

    unique_pairs_params = {}
    unique_bonds_params = {}
    for type, infos in params.items():
        if type == "pairs":
            for key, values in infos.items():
                elems = key.split("-")
                assert len(elems) == 2, "The number of elements in pairs must be equal to 2"

                elem1s = {k: v for k, v in elem_mapping.items() if fnmatch.fnmatch(k, elems[0])}
                elem2s = {k: v for k, v in elem_mapping.items() if fnmatch.fnmatch(k, elems[1])}
                
                for elem1, type1 in elem1s.items():
                    for elem2, type2 in elem2s.items():
                        unique_pair = (min(type1, type2), max(type1, type2))
                        if unique_pair in unique_pairs_params.keys():
                            continue
                        alter_key = f"{elem2}-{elem1}"
                        if alter_key in params.keys() and len(values) == 2:
                            eplison = math.sqrt(values[0]*params[alter_key][0])
                            sigma = (values[1]+params[alter_key][1]) / 2

                            unique_pairs_params[unique_pair] = [eplison, sigma]
                        else:
                            unique_pairs_params[unique_pair] = values
        elif type == "bonds":
            for key, values in infos.items():
                bond_list = parse_range_list(key)
                for bond_id in bond_list:
                    unique_bonds_params[bond_id] = values

    pairs_params = unique_pairs_params
    bonds_params = unique_bonds_params

    # Read and process input files
    with open(input_file) as f:
        lines = f.readlines()

    # A regular expression used to match pair_coeff rows
    pair_coeff_pattern = re.compile(
        r"^pair_coeff\s+(\d+)\s+(\d+)\s+([a-zA-Z/_]+)(.*)$"
    )

    bond_coeff_pattern = re.compile(
        r"^bond_coeff\s+(\d+)\s+([\d\.Ee+-]+)\s+([\d\.Ee+-]+)"
    )

    new_lines = []
    for line in lines:
        # Match pair_coeff rows
        match = pair_coeff_pattern.match(line.strip())        
        if match:
            type_i = int(match.group(1))
            type_j = int(match.group(2))
            style = match.group(3)
            rest = match.group(4).rsplit('#')[0].split()

            # Find the corresponding parameters
            key = (min(type_i, type_j), max(type_i, type_j))
            if key in pairs_params.keys():
                params_values = " ".join(map(str, pairs_params[key]))
                rest_values = " ".join(map(str, rest[len(pairs_params[key]):]))
                if len(pairs_params[key]) > len(rest):
                    new_line = f"pair_coeff {type_i} {type_j} {style} {rest_values}\n"
                else:
                    new_line = f"pair_coeff {type_i} {type_j} {style} {params_values} {rest_values}\n"
                new_lines.append(new_line)
                continue         
        
        match = bond_coeff_pattern.match(line.strip())    
        if match:
            type_i = int(match.group(1))
            params_values = float(match.group(2))
            rest_values = match.group(3).rsplit('#')[0].split()
            if type_i in bonds_params.keys():
                params_values = " ".join(map(str, bonds_params[type_i]))
                rest_values = " ".join(map(str, rest_values))
                new_line = f"bond_coeff {type_i} {params_values} {rest_values}\n"
                new_lines.append(new_line)
                continue    

        new_lines.append(line)

    # Write the output file
    with open(output_file, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
	lmp_file = sys.argv[1]
	out_file = sys.argv[2]
	params_file = sys.argv[3]
	replace_lammps_params(
		input_file=lmp_file,
		output_file=out_file,
		params_file=params_file
	)
