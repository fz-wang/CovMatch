"""Generate per-NCAA Rosetta Match flags and a Bash runner."""
import argparse
import math
import os
from pathlib import Path
import shlex
from ncaa import PARAMETERS, NCAA_name_lib, read_entry


def select_names(value):
    names = set()
    for token in value.upper().split(','):
        token = token.strip()
        if token == 'ALL':
            names.update(NCAA_name_lib)
        elif token in NCAA_name_lib:
            names.add(token)
        elif token in {'LA','LC','LM','LO','LP'}:
            names.update(n for n in NCAA_name_lib if n.startswith(token))
        else:
            raise ValueError(f'Unknown NCAA/group {token}; use a name, LA/LC/LM/LO/LP, or ALL.')
    return sorted(names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('-s', '--pdb-file', type=Path)
    source.add_argument('-l', '--pdb-list', type=Path)
    p.add_argument('--pos', required=True, nargs='+', type=Path)
    p.add_argument('-n', '--ncaa', required=True)
    p.add_argument('-db', '--parameters', type=Path, default=PARAMETERS)
    p.add_argument('--match-bin', default=os.environ.get('ROSETTA_MATCH_BIN','match.default.linuxgccrelease'))
    p.add_argument('--rosetta-database', type=Path)
    p.add_argument('-b', '--bump-tolerance', type=float, default=0.)
    p.add_argument('-bin', '--bin-size', default='0.3,5.0')
    p.add_argument('-g', '--matches-per-group', type=int)
    p.add_argument('--format', choices=['PDB','CloudPDB'], default='PDB')
    p.add_argument('-o', '--output-dir', type=Path, default=Path('match_jobs'))
    a = p.parse_args()
    names = select_names(a.ncaa)
    euclid,euler = map(float,a.bin_size.split(','))
    if not all(math.isfinite(v) and v>0 for v in [euclid,euler]) or not math.isclose(180/euler,round(180/euler)):
        p.error('Use positive bin sizes; the Euler size must divide 180.')
    count = a.matches_per_group if a.matches_per_group is not None else (10 if a.pdb_list else 30)
    if count < 1 or not math.isfinite(a.bump_tolerance) or a.bump_tolerance < 0:
        p.error('Use positive matches-per-group and nonnegative bump tolerance.')
    scaffolds = [a.pdb_file.resolve()] if a.pdb_file else [
        (a.pdb_list.resolve().parent/line.strip()).resolve()
        for line in a.pdb_list.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not scaffolds:
        p.error('The PDB list is empty.')
    if len({p.stem for p in scaffolds}) != len(scaffolds) or len({p.stem for p in a.pos}) != len(a.pos):
        p.error('Input filenames must have unique stems.')
    for path in [*scaffolds,*a.pos]:
        if not path.is_file(): p.error(f'Input not found: {path}')
    root = a.parameters.resolve()
    entries = [read_entry(n,root) for n in names]
    prepared=[]
    for entry in entries:
        params=[root/'warheads'/f'{entry.warhead}.params',root/'targets'/f'{entry.target}.params',root/'stubs'/f'{entry.stub}.params']
        for path in params:
            if not path.is_file(): p.error(f'Parameter not found: {path}')
        for pos in a.pos:
            for scaffold in scaffolds:
                name=f'{scaffold.stem}_{entry.ncaa_name}_{pos.stem}'
                flags=['-ex1','-ex2','-use_input_sc','-extrachi_cutoff 0',
                       '-extra_res_fa '+' '.join('"'+str(x)+'"' for x in params),
                       f'-match:lig_name {entry.warhead}',
                       f'-match:geometric_constraint_file "{root/"constraints"/(entry.ncaa_name+".cst")}"',
                       f'-match:scaffold_active_site_residues_for_geomcsts "{pos.resolve()}"',
                       f'-match:output_format {a.format}', '-match_grouper SameSequenceGrouper',
                       '-match:filter_colliding_upstream_residues','-match:filter_upstream_downstream_collisions',
                       f'-match:bump_tolerance {a.bump_tolerance}',
                       '-enumerate_ligand_rotamers true','-only_enumerate_non_match_redundant_ligand_rotamers true',
                       f'-match:euclid_bin_size {euclid}',f'-match:euler_bin_size {euler}',
                       '-consolidate_matches', f'-output_matches_per_group {count}',
                       '-ignore_zero_occupancy false','-dynamic_grid_refinement',
                       '-match:grouper_downstream_rmsd 1.5','-output_virtual true']
                if a.rosetta_database:
                    flags.append(f'-database "{a.rosetta_database.resolve()}"')
                prepared.append((name,scaffold,'\n'.join(flags)+'\n'))
    out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    commands=['#!/usr/bin/env bash','set -euo pipefail','']
    for name,scaffold,flags in prepared:
        job=out/name; job.mkdir(exist_ok=True)
        flagfile=job/'match.flags'; flagfile.write_text(flags,encoding='utf-8',newline='\n')
        command=shlex.join([a.match_bin,'@'+str(flagfile),'-s',str(scaffold)])
        commands.append(f'(cd {shlex.quote(str(job))} && {command} > match.log 2>&1)')
    runner=out/'run_all_matches.sh'
    runner.write_text('\n'.join(commands)+'\n',encoding='utf-8',newline='\n')
    print(f'{len(prepared)} Match jobs: {runner}')


if __name__ == '__main__':
    main()
