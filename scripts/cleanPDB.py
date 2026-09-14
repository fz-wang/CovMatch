"""Remove water and metal records from a single PDB model."""
import argparse
from pathlib import Path

METALS = {'LI', 'NA', 'K', 'RB', 'CS', 'MG', 'CA', 'SR', 'BA', 'AL', 'MN',
          'FE', 'CO', 'NI', 'CU', 'ZN', 'CD', 'HG'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-p', '--pdb-file', required=True, type=Path)
    p.add_argument('-o', '--output', type=Path)
    p.add_argument('-m', '--keep-metals', action='store_true')
    p.add_argument('-w', '--keep-waters', action='store_true')
    a = p.parse_args()
    output = a.output or a.pdb_file.with_name(a.pdb_file.stem + '_clean.pdb')
    if output.resolve() == a.pdb_file.resolve():
        p.error('Choose an output different from the input.')
    lines = a.pdb_file.read_text().splitlines()
    removed = set()
    kept = []
    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            water = line[17:20].strip() in {'HOH', 'WAT', 'DOD'}
            metal = line.startswith('HETATM') and line[76:78].strip().upper() in METALS
            if (water and not a.keep_waters) or (metal and not a.keep_metals):
                removed.add(int(line[6:11]))
                continue
        kept.append(line)
    result = []
    for line in kept:
        if line.startswith('CONECT'):
            numbers = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
            if not numbers or numbers[0] in removed:
                continue
            numbers = [n for n in numbers if n not in removed]
            if len(numbers) < 2:
                continue
            line = 'CONECT' + ''.join(f'{n:5d}' for n in numbers)
        result.append(line)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(result) + '\n', encoding='utf-8', newline='\n')
    print(output)


if __name__ == '__main__':
    main()
