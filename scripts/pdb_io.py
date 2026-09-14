"""PDB atom records shared by the CovMatch command-line scripts."""
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class Atom:
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    charge: str = ''

    @classmethod
    def from_pdb_line(cls, line):
        return cls(line[:6].strip(), int(line[6:11]), line[12:16].strip(),
                   line[17:20].strip(), line[21:22].strip(), int(line[22:26]),
                   line[26:27].strip(), float(line[30:38]), float(line[38:46]),
                   float(line[46:54]), float(line[54:60].strip() or 1),
                   float(line[60:66].strip() or 0),
                   line[76:78].strip() or line[12:16].strip().lstrip('0123456789')[0],
                   line[78:80].strip())

    def residue_key(self):
        return self.chain, self.resseq, self.icode

    def is_hydrogen(self):
        return self.element.upper() in {'H', 'D'}

    def to_pdb_line(self):
        name = self.name if len(self.name) == 4 or len(self.element) == 2 else ' ' + self.name
        return (f'{self.record:<6}{self.serial:5d} {name:<4} {self.resname:>3} '
                f'{self.chain:1}{self.resseq:4d}{self.icode:1}   '
                f'{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}'
                f'{self.occupancy:6.2f}{self.bfactor:6.2f}          '
                f'{self.element:>2}{self.charge:>2}\n')


def parse_atoms(lines):
    atoms = []
    keys = set()
    for line in lines:
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        if line[16:17].strip():
            raise ValueError('Resolve alternate locations before running CovMatch.')
        atom = Atom.from_pdb_line(line)
        key = (*atom.residue_key(), atom.name)
        if key in keys:
            raise ValueError(f'Duplicate atom {key}; use one PDB model.')
        keys.add(key)
        atoms.append(atom)
    return atoms


def read_atoms(path):
    return parse_atoms(Path(path).read_text().splitlines())


def residue_order(atoms):
    result = []
    names = {}
    for atom in atoms:
        key = atom.residue_key()
        if key in names and names[key] != atom.resname:
            raise ValueError(f'Multiple residue names at {key}')
        if key not in names:
            result.append(key)
            names[key] = atom.resname
    return result


def write_atoms(path, atoms):
    Path(path).write_text(''.join(replace(a, serial=i).to_pdb_line()
                                for i, a in enumerate(atoms, 1)) + 'END\n',
                          encoding='utf-8', newline='\n')
