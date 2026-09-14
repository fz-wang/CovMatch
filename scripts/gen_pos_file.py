"""Select peptide sites by CA distance and write Rosetta Match position files."""
import argparse
import math
import re
from pathlib import Path
from pdb_io import read_atoms, residue_order


def target_key(value):
    m = re.fullmatch(r'([^:\s]):(-?\d+)([A-Za-z]?)', value)
    if m:
        return m[1], int(m[2]), m[3]
    m = re.fullmatch(r'(-?\d+)([A-Za-z0-9])', value)
    if m:
        return m[2], int(m[1]), ''
    raise ValueError(f'Invalid site {value}; use A:163 or 163A.')


def positions(atoms, target, binder, cutoff):
    order = {key: i for i, key in enumerate(residue_order(atoms), 1)}
    ca = {a.residue_key(): a for a in atoms if a.name == 'CA'}
    if target not in ca:
        raise ValueError(f'No CA at target {target}')
    t = ca[target]
    sites = [order[key] for key, a in ca.items() if key[0] == binder
             and key != target and math.dist((t.x,t.y,t.z), (a.x,a.y,a.z)) <= cutoff]
    if not sites:
        raise ValueError(f'No binder CA atoms within {cutoff} A of {target}')
    return order[target], sites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-s', '-p', '--pdb-file', required=True, type=Path)
    p.add_argument('-nu', '--nucleophilic-residue', required=True)
    p.add_argument('-b', '--binder-chain', required=True)
    p.add_argument('-d', '--distance', default=12., type=float)
    p.add_argument('-o', '--output-dir', type=Path)
    a = p.parse_args()
    if len(a.binder_chain) != 1 or not math.isfinite(a.distance) or a.distance <= 0:
        p.error('Use a single binder chain ID and a positive finite distance.')
    atoms = read_atoms(a.pdb_file)
    output = a.output_dir or a.pdb_file.parent
    prepared = []
    for value in a.nucleophilic_residue.split(','):
        key = target_key(value.strip())
        target, sites = positions(atoms, key, a.binder_chain, a.distance)
        filename = f'{a.pdb_file.stem}_{key[1]}{key[2]}{key[0]}.pos'
        prepared.append((output/filename, f'N_CST 2\n1: {target}\n2: ' + ' '.join(map(str,sites)) + '\n'))
    output.mkdir(parents=True, exist_ok=True)
    for path, text in prepared:
        path.write_text(text, encoding='utf-8', newline='\n')
        print(path)


if __name__ == '__main__':
    main()
