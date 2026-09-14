"""Atom mappings and Match geometry for the distributed L-NCAA parameters."""
from dataclasses import dataclass
from pathlib import Path
import re

PARAMETERS = Path(__file__).resolve().parents[1] / 'parameters'
# Warhead anchor atoms in the full residue: amide N, carbonyl C, next C.
ANCHORS = {
    'LA5': ('NG','CD','CE2'), 'LA6': ('ND','CE','CZ2'),
    'LA7': ('NE','CZ','CT2'), 'LA8': ('NZ','CT','CI2'),
    'LC4': ('NG','CD','CE2'), 'LC5': ('ND','CE','CZ2'),
    'LC6': ('NE','CZ','CT2'), 'LC7': ('NZ','CT','CI2'),
    'LM1': ('NZ1','CT','CI2'), 'LM2': ('NZ1','CT','CI2'),
    'LM3': ('NT','CI','CK2'), 'LM4': ('NT','CI','CK2'),
    'LO1': ('NE1','CZ2','CT2'), 'LO2': ('NE1','CZ2','CT2'),
    'LO3': ('NZ1','CT','CI2'), 'LO4': ('NZ1','CT','CI2'),
    'LP1': ('NT','CI','CK2'), 'LP2': ('NT','CI','CK2'),
    'LP3': ('NI','CK','CL2'), 'LP4': ('NI','CK','CL2'),
}
target_conj_atom = {'CYW': ('SG','CB'), 'CYF': ('SG','CB')}


def normalize_target_name(name):
    return {'CYW':'CYF'}.get(name, name)


@dataclass
class NCAA:
    ncaa_name: str
    warhead: str
    stub: str
    target: str
    conj: str
    subconj: str
    warhead_2_ncaa: dict
    cst_values: tuple
    stub_2_ncaa: dict


def parameter_atoms(path):
    rows = [line.split() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    types = {r[1]: r[2] for r in rows if r[0] == 'ATOM'}
    bonds = {a: set() for a in types}
    for r in rows:
        if r[0] in {'BOND','BOND_TYPE'}:
            bonds[r[1]].add(r[2]); bonds[r[2]].add(r[1])
    return rows, types, bonds


def read_entry(name, root=None):
    if name not in ANCHORS:
        raise ValueError(f'Unknown NCAA: {name}')
    root = Path(root or PARAMETERS)
    text = (root/'constraints'/f'{name}.cst').read_text()
    residues = re.findall(r'residue3:\s*(\w+)', text)
    warhead, target, repeated, stub = residues
    if warhead != repeated:
        raise ValueError(f'Inconsistent warhead in {name}.cst')
    rows, types, bonds = parameter_atoms(root/'final'/f'{name}.params')
    anchors = ('N1','C1','C2') if warhead == '2CW' else ('N1','C2','C1')
    mapping = dict(zip(anchors, ANCHORS[name]))
    for atom, neighbor, element in [('O1',anchors[1],'O'), ('H1','N1','H')]:
        matches = [n for n in bonds[mapping[neighbor]] if types[n].startswith(element)]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous {name} mapping for {atom}: {matches}')
        mapping[atom] = matches[0]
    if warhead == '3AW':
        matches = [n for n in bonds[mapping['C1']] if n not in mapping.values()
                   and types[n].startswith('C')]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous terminal carbon for {name}: {matches}')
        mapping['C3'] = matches[0]
    conj = next(r[1] for r in rows if r[0] == 'CONNECT')
    virtual = next(r for r in rows if r[:2] == ['ICOOR_INTERNAL','CONN3'])
    assert virtual[5] == conj
    geometry = tuple(float(re.search(rf'CONSTRAINT::\s*{field}:\s*([-\d.]+)', text)[1])
                     for field in ['distanceAB','angle_A','angle_B'])
    stub_map = {}
    if name in {'LM1','LM2','LO3','LO4'}:
        stub_map = {'CZ':'CZ2', '1HZ':'1HZ2'}
    elif name in {'LO1','LO2'}:
        stub_map = {'CE1':'CE2', 'CE2':'CE3', 'CZ':'CZ1'}
    return NCAA(name,warhead,stub,target,conj,virtual[6],mapping,geometry,stub_map)


def find_mapping(warhead, stub, target, ncaa_name=None):
    choices = [read_entry(n) for n in ([ncaa_name] if ncaa_name else sorted(ANCHORS))]
    choices = [e for e in choices if e.warhead == warhead and e.stub == stub
               and normalize_target_name(e.target) == normalize_target_name(target)]
    if len(choices) > 1:
        raise ValueError('Several NCAAs share this stub; provide --ncaa or use the original Match filename.')
    return choices[0] if choices else None


NCAA_name_lib = set(ANCHORS)
warhead_lib = {'2CW','3AW'}
stub_lib = {'L1A','L2A','L3A','L4A','LFB','LMY','LOY','LPY'}
