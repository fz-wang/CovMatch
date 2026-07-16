import os
import argparse
import subprocess
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
import time
import shutil
import math
import shlex
import tempfile
import getpass

gaussian_path = "/home/fzwang/install/gaussian16/g16/g16"
rosetta_path = "/home/fzwang/rosetta_bin_linux_2020.25.61318_bundle"
rosetta_python = os.environ.get("ROSETTA_PYTHON", "/home/fzwang/miniconda3/envs/py2/bin/python")

COVALENT_TARGETS = ("CYS", "LYS", "HIS", "TYR")
CAP_IGNORE_ATOM_IDS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
POLY_LOWER_ATOM_ID = 1
POLY_UPPER_ATOM_ID = 7
POLY_N_ATOM_ID = 13
POLY_CA_ATOM_ID = 14
POLY_C_ATOM_ID = 15
POLY_O_ATOM_ID = 16

# Query order is ACE methyl/carbonyl/O, backbone N/CA/C/O, then NME N/methyl.
CAPPED_BACKBONE_QUERY = Chem.MolFromSmarts(
    "[CH3:1][C:2](=[O:3])[N:4][C;X4:5][C:6](=[O:7])[N:8][CH3:9]"
)

def fail(message):
    banner = "\n" + "=" * 80 + "\n[FATAL ERROR] " + message + "\n" + "=" * 80
    raise RuntimeError(banner)

def run_command(command, label, capture_output=False):
    print(f"[RUN] {label}: {command}")
    result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
    if result.returncode != 0:
        details = ""
        if capture_output:
            details = f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        fail(f"{label} failed with exit code {result.returncode}.{details}")
    return result


def find_capped_backbone_match(mol):
    matches = mol.GetSubstructMatches(CAPPED_BACKBONE_QUERY, uniquify=True)
    if len(matches) > 1:
        fail(f"Found {len(matches)} ACE-NCAA-NME backbone matches; expected exactly one.")
    return matches[0] if matches else None


def prepare_input_smiles(smiles, input_form):
    """Return an ACE-NCAA-NME SMILES while accepting free or already capped inputs."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to parse SMILES: '{smiles}'")
    capped_match = find_capped_backbone_match(mol)

    if input_form == "capped" and capped_match is None:
        raise ValueError("--input-form capped was used, but no ACE-NCAA-NME backbone was found.")
    if input_form == "free" and capped_match is not None:
        raise ValueError("--input-form free was used, but the input already contains ACE and NME caps.")

    if capped_match is not None:
        print("Detected an existing ACE-NCAA-NME capped model; skipping automatic capping.")
        return Chem.MolToSmiles(mol, canonical=False)
    return process_smiles(smiles)


def reorder_capped_molecule(mol):
    """Put capped backbone heavy atoms first so legacy atom IDs 1-16 remain valid."""

    match = find_capped_backbone_match(mol)
    if match is None:
        fail("Cannot establish ACE-NCAA-NME atom order after capping.")
    backbone_heavy = list(match)
    remaining_heavy = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1 and atom.GetIdx() not in backbone_heavy
    ]
    hydrogens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1]
    order = backbone_heavy + remaining_heavy + hydrogens
    if len(order) != mol.GetNumAtoms() or len(set(order)) != len(order):
        fail("Failed to construct a unique capped-model atom order.")
    return Chem.RenumberAtoms(mol, order)

# 定义乙酰化和甲氨基化的SMARTS模式
acetylation_smarts = '[N:1][C:2][C:3](=[O:4])>>[C:5][C:7](=[O:6])[N:1][C:2][C:3](=[O:4])'    # 乙酰化，为主链N连上乙酰基[C:5][C:7](=[O:6])
amidation_smarts = '[C:1][C:2](=[O:3])[O:4]>>[C:1][C:2]([N:5][C:6])(=[O:3])'                 # 甲氨基化，为羧基主链C连上甲氨基([N:5][C:6])(=[O:3]。注意这里reactant里需要将单键O[O:4]标注出来，不然会报错

#根据smiles的电荷信息计算体系净电荷值
def calculate_system_charge(smiles_filepath):
    with open(smiles_filepath, 'r') as file:
        smiles = file.read().strip()
    
    negative_count = smiles.count('-')
    positive_count = smiles.count('+')
    
    system_charge = positive_count - negative_count
    return system_charge
    
def process_smiles(smiles):
    # 识别输入的smiles是否可解析
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unable to parse SMILES: '{smiles}'")

    # 创建乙酰化和甲氨基化的反应
    rxn_acetylation = AllChem.ReactionFromSmarts(acetylation_smarts)
    rxn_amidation = AllChem.ReactionFromSmarts(amidation_smarts)

    # 应用甲氨基化反应
    products_amidation = rxn_amidation.RunReactants((mol,))
    if not products_amidation:
        raise ValueError("C-terminal amidation failed")

    product_amidation = products_amidation[0][0]

    # 应用乙酰化反应
    products_acetylation = rxn_acetylation.RunReactants((product_amidation,))
    if not products_acetylation:
        raise ValueError("N-terminal acetylation failed")

    product_acetylation = products_acetylation[0][0]

    # 获取封端后的分子的 SMILES
    final_smiles = Chem.MolToSmiles(product_acetylation, canonical=False)
    print(final_smiles)
    return final_smiles

def smiles_to_pdb(smiles_list, names_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(names_list) < len(smiles_list):
        names_list.extend(['UAA'] * (len(smiles_list) - len(names_list)))
    elif len(names_list) > len(smiles_list):
        raise ValueError("The number of provided names exceeds the number of SMILES strings.")

    generated_files = []
    for i, (smiles, name) in enumerate(zip(smiles_list, names_list)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"SMILES字符串 '{smiles}' 无法解析")
            continue
        mol = reorder_capped_molecule(mol)
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != 0:
            fail(f"RDKit could not embed a 3D conformer for {name}.")
        
        # 生成PDB文件名
        pdb_filename = os.path.join(output_dir, f'{name}_{i+1}.pdb')
        
        # 写入PDB文件并添加注释(该NCAA封端后的smiles，便于使用者检查)
        with open(pdb_filename, 'w') as pdb_file:
            pdb_file.write(f"REMARK {smiles}\n")
            pdb_file.write(Chem.rdmolfiles.MolToPDBBlock(mol))
        
        generated_files.append(pdb_filename)
        print(f"生成 {pdb_filename}")

        # 读取并打印PDB文件的内容
        with open(pdb_filename, 'r') as pdb_file_print:
            pdb_content = pdb_file_print.read()
            print(pdb_content)

    return generated_files

def process_pdb_file(filepath, n_value, output_filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 过滤掉不是以 "HETATM" 或 "ATOM  " 开头的行
    lines = [line for line in lines if line.startswith("HETATM") or line.startswith("ATOM  ")]

    # 字典存储需要进行调整的行
    line_dict = { "C2": None, "O1": None, "C1": None, "H1": None, "H2": None, "H3": None,
                  "N2": None, "C5": None, "H6": None, "H7": None, "H8": None, "H9": None,
                  "N1": None, "C3": None, "C4": None, "O2": None, "H4": None, "H5": None }
    other_lines = []

    for line in lines:
        atom_name = line[12:16].strip()
        if atom_name in line_dict:
            line_dict[atom_name] = line
        else:
            other_lines.append(line)

    # 按指定顺序重新排列行
    reordered_lines = []
    for key in ["C2", "O1", "C1", "H1", "H2", "H3", "N2", "C5", "H6", "H7", "H8", "H9", "N1", "C3", "C4", "O2"]:
        if line_dict[key] is not None:
            reordered_lines.append(line_dict[key])

    reordered_lines.extend(other_lines)

    #后置主链H原子
    if line_dict["H4"] is not None:
        reordered_lines.append(line_dict["H4"])

    if line_dict["H5"] is not None:
        reordered_lines.append(line_dict["H5"])

    # 修改氨基酸名称，指认出ACE和NME的部分
    for i in range(len(reordered_lines)):
        if i < 6:
            reordered_lines[i] = reordered_lines[i][:17] + "ACE" + reordered_lines[i][20:]
        elif i < 12:
            reordered_lines[i] = reordered_lines[i][:17] + "NME" + reordered_lines[i][20:]
        else:
            reordered_lines[i] = reordered_lines[i][:17] + n_value + reordered_lines[i][20:]

    # 添加 "END" 行
    reordered_lines.append("END\n")

    # 写回新文件
    with open(output_filepath, 'w') as file:
        file.writelines(reordered_lines)

# 遍历所有pdb，对其执行重排操作
def process_files(filepaths, n_values, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (filepath, n_value) in enumerate(zip(filepaths, n_values)):
        output_filepath = os.path.join(output_dir, f'{n_value}_{i+1}.pdb')
        process_pdb_file(filepath, n_value, output_filepath)
        print(f"Processed {filepath}")

        # 读取并打印PDB文件的内容
        with open(output_filepath, 'r') as output_print:
            output_content = output_print.read()
            print(output_content)

#定义二面角设置函数，应用于adjust_dihedrals_in_pdb_files函数的二面角调整，在发生错误时提供错误信息(如果该步出现报错则很有可能是上一步原子重排没有排好)
def set_dihedral_angle(mol, atom_indices, angle):
    try:
        SetDihedralDeg(mol.GetConformer(), *atom_indices, angle)
    except Exception as e:
        print(f"Error setting dihedral angle for atoms {atom_indices} to {angle} degrees: {e}")
        
def adjust_dihedrals_in_pdb_files(filepaths):
    for pdb_path in filepaths:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            print(f"Unable to load PDB file: {pdb_path}")
            continue
        
        # 获取原子索引
        atom_indices = {atom.GetPDBResidueInfo().GetName().strip(): atom.GetIdx() for atom in mol.GetAtoms()}

        # 检查 N1 原子连接的非氢原子数目
        N1_atom = mol.GetAtomWithIdx(atom_indices['N1'])
        non_h_neighbors = [nbr for nbr in N1_atom.GetNeighbors() if nbr.GetSymbol() != 'H']
        
        #根据N原子连接的非H原子数判断对象是peptide还是peptoid，并根据情况设置其对应的优势二面角
        if len(non_h_neighbors) == 2:
            # peptide
            set_dihedral_angle(mol, [atom_indices['C2'], atom_indices['N1'], atom_indices['C3'], atom_indices['C4']], -150.0)
            set_dihedral_angle(mol, [atom_indices['N1'], atom_indices['C3'], atom_indices['C4'], atom_indices['N2']], 150.0)
        elif len(non_h_neighbors) == 3:
            # peptoid
            set_dihedral_angle(mol, [atom_indices['C2'], atom_indices['N1'], atom_indices['C3'], atom_indices['C4']], -120.0)
            set_dihedral_angle(mol, [atom_indices['N1'], atom_indices['C3'], atom_indices['C4'], atom_indices['N2']], 90.0)
        else:
            print(f"Unexpected number of non-hydrogen neighbors for N1 in {pdb_path}")
            continue
        
        # 写回调整后的PDB文件
        with open(pdb_path, 'w') as f:
            f.write(Chem.MolToPDBBlock(mol))

        print(f"Adjusted dihedrals in {pdb_path}")

#过滤掉CONECT行，键连信息的存在有时会导致antechamber转化gjf文件时报错
def remove_conect_lines_from_pdb(pdb_filepath):
    with open(pdb_filepath, 'r') as f:
        lines = f.readlines()

    # 过滤掉以 "CONECT" 开头的行
    lines = [line for line in lines if not line.startswith("CONECT")]

    # 写回新文件
    with open(pdb_filepath, 'w') as f:
        f.writelines(lines)

    print(f"Removed CONECT lines from {pdb_filepath}")

#提交gaussian作业
def gaussian_environment_prefix():
    gaussian_dir = os.path.dirname(gaussian_path)
    gaussian_root = os.path.dirname(gaussian_dir)
    gaussian_profile = os.path.join(gaussian_dir, 'bsd', 'g16.profile')
    if not os.path.exists(gaussian_path):
        fail(f"Gaussian executable is missing: {gaussian_path}")
    if not os.path.exists(gaussian_profile):
        fail(f"Gaussian environment profile is missing: {gaussian_profile}")
    scratch_dir = os.environ.get(
        'RG_GAUSSIAN_SCRDIR',
        os.path.join(tempfile.gettempdir(), getpass.getuser(), 'gaussian16'),
    )
    os.makedirs(scratch_dir, exist_ok=True)
    return (
        f"export g16root={shlex.quote(gaussian_root)}; "
        f"export GAUSS_SCRDIR={shlex.quote(scratch_dir)}; "
        f". {shlex.quote(gaussian_profile)};"
    )


def gaussian_command(gjf_filepath, log_filepath):
    return (
        f"{gaussian_environment_prefix()} "
        f"{shlex.quote(gaussian_path)} < {shlex.quote(gjf_filepath)} "
        f"> {shlex.quote(log_filepath)}"
    )


def submit_gaussian_job(gjf_filepath):
    log_filepath = gjf_filepath.replace('.gjf', '.log')
    command = gaussian_command(gjf_filepath, log_filepath)
    process = subprocess.Popen(command, shell=True)
    print(f"Submitted Gaussian job for {gjf_filepath}")
    return log_filepath, process


def gaussian_resource_values():
    cpu_override = os.environ.get('RG_GAUSSIAN_CPUS')
    memory_override = os.environ.get('RG_GAUSSIAN_MEMORY')

    if cpu_override:
        cpus = int(cpu_override)
        if cpus < 1:
            fail('RG_GAUSSIAN_CPUS must be a positive integer.')
    else:
        cpus = min(os.cpu_count() or 1, 8)

    if memory_override:
        memory = memory_override
    else:
        total_gib = None
        try:
            with open('/proc/meminfo', 'r') as meminfo:
                for line in meminfo:
                    if line.startswith('MemTotal:'):
                        total_kib = int(line.split()[1])
                        total_gib = total_kib / (1024 * 1024)
                        break
        except (OSError, ValueError):
            pass
        memory_gib = max(2, min(32, int((total_gib or 8) * 0.60)))
        memory = f'{memory_gib}GB'

    return memory, cpus


def set_gaussian_resources(gjf_filepath):
    memory, cpus = gaussian_resource_values()
    with open(gjf_filepath, 'r') as gjf_file:
        lines = gjf_file.readlines()
    lines = [
        line for line in lines
        if not line.lower().startswith('%mem=')
        and not line.lower().startswith('%nprocshared=')
    ]
    if lines and lines[0].strip() == '--Link1--':
        # Antechamber emits this separator even for a single-link GJF. At the
        # beginning of a file it prevents Gaussian from finding the route card.
        lines.pop(0)
    lines[0:0] = [f'%mem={memory}\n', f'%nprocshared={cpus}\n']
    with open(gjf_filepath, 'w') as gjf_file:
        gjf_file.writelines(lines)
    print(f"Configured Gaussian resources for {gjf_filepath}: memory={memory}, cpus={cpus}")
    
def modify_gjf_file(gjf_filepath, system_charge):
    with open(gjf_filepath, 'r') as file:
        lines = file.readlines()

    # 检索并修改电荷行，为gjf文件提供正确的电荷信息
    for i in range(len(lines)):
        if lines[i].strip() == "0   1":
            lines[i] = f"{system_charge}   1\n"
            break

    # 在最后一行非空行后添加固定二面角信息，以确保结构优化时Phi，Psi角固定不动
    non_empty_lines = [line for line in lines if line.strip()]
    last_non_empty_line_index = lines.index(non_empty_lines[-1])
    lines.insert(last_non_empty_line_index + 1, '\n13 14 15 7 F\n1 13 14 15 F\n\n\n\n')

    with open(gjf_filepath, 'w') as file:
        file.writelines(lines)

    set_gaussian_resources(gjf_filepath)

    print(f"Modified {gjf_filepath}")


def pdb_element_sequence(pdb_filepath):
    elements = []
    with open(pdb_filepath, 'r') as pdb_file:
        for line in pdb_file:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            element = line[76:78].strip()
            if not element:
                fail(f"Missing PDB element field in {pdb_filepath}: {line.rstrip()}")
            elements.append(element.capitalize())
    return elements


def gjf_element_sequence(gjf_filepath):
    with open(gjf_filepath, 'r') as gjf_file:
        lines = gjf_file.readlines()
    charge_line = None
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2:
            try:
                int(parts[0])
                int(parts[1])
            except ValueError:
                continue
            charge_line = index
            break
    if charge_line is None:
        fail(f"Cannot find charge/multiplicity line in {gjf_filepath}.")

    elements = []
    for line in lines[charge_line + 1:]:
        parts = line.split()
        if not parts:
            if elements:
                break
            continue
        if len(parts) < 4:
            fail(f"Unexpected Gaussian coordinate line in {gjf_filepath}: {line.rstrip()}")
        elements.append(parts[0].capitalize())
    return elements


def validate_gjf_elements(pdb_filepath, gjf_filepath):
    pdb_elements = pdb_element_sequence(pdb_filepath)
    gjf_elements = gjf_element_sequence(gjf_filepath)
    if pdb_elements != gjf_elements:
        mismatches = [
            (index + 1, pdb_element, gjf_element)
            for index, (pdb_element, gjf_element) in enumerate(zip(pdb_elements, gjf_elements))
            if pdb_element != gjf_element
        ]
        fail(
            f"PDB/GJF element mismatch for {gjf_filepath}: "
            f"{len(pdb_elements)} PDB atoms versus {len(gjf_elements)} GJF atoms; "
            f"first mismatches {mismatches[:5]}."
        )
    print(f"Validated {len(pdb_elements)} PDB/GJF element identities for {gjf_filepath}.")

#生成Gaussian输入文件
def generate_gaussian_input(filepaths, output_dir, system_charge, submit=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submitted_jobs = []
    for pdb_path in filepaths:
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        gjf_filename = os.path.join(output_dir, f'{base_name}.gjf')
        log_filename = os.path.join(output_dir, f'{base_name}.log')

        if os.path.exists(log_filename) and is_gaussian_job_completed(log_filename):
            print(f"Found completed Gaussian log {log_filename}; skipping Gaussian optimization.")
            continue

        command = f'antechamber -i {pdb_path} -fi pdb -o {gjf_filename} -fo gcrt -gk "# opt=(modredundant,loose) b3lyp/6-31+g(d) scrf=(solvent=water) empiricaldispersion=gd3bj"'
        run_command(command, f"Generate Gaussian input for {pdb_path}")

        modify_gjf_file(gjf_filename, system_charge)

        # Preserve real halogens. Fail loudly if antechamber changed atom identities.
        validate_gjf_elements(pdb_path, gjf_filename)

        print(f"Generated {gjf_filename}")

        if submit:
            submitted_jobs.append(submit_gaussian_job(gjf_filename))
        else:
            print(f"Skipped Gaussian submission for {gjf_filename}")
    return submitted_jobs

#检查函数，用以监督Gaussian作业是否结束运行
def is_gaussian_job_completed(log_filepath):
    with open(log_filepath, 'r') as file:
        lines = file.readlines()
    return any("Normal termination" in line for line in lines)

def is_resp_gaussian_job_completed(log_filepath):
    """Return True only for a completed RESP charge Gaussian log."""
    if not os.path.exists(log_filepath):
        return False
    with open(log_filepath, 'r') as file:
        text = file.read()
    return (
        "Normal termination" in text
        and ("Pop=MK" in text or "HF/6-31G" in text)
    )


def heavy_neighbors(atom):
    return [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() != 1]


def component_without_bond(mol, start_idx, atom_a, atom_b):
    blocked = {frozenset((atom_a, atom_b))}
    visited = set()
    stack = [start_idx]
    while stack:
        atom_idx = stack.pop()
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if frozenset((atom_idx, neighbor_idx)) in blocked:
                continue
            if neighbor_idx not in visited:
                stack.append(neighbor_idx)
    return visited


def covalent_template_candidates(mol, target_type):
    """Return (target nucleophile, NCAA connection atom, target heavy atoms)."""

    candidates = []
    if target_type == "CYS":
        query = Chem.MolFromSmarts("[S:1]-[C:2]")
        for sulfur_idx, carbon_idx in mol.GetSubstructMatches(query, uniquify=True):
            sulfur = mol.GetAtomWithIdx(sulfur_idx)
            carbon = mol.GetAtomWithIdx(carbon_idx)
            if len(heavy_neighbors(carbon)) != 1:
                continue
            sulfur_heavy = heavy_neighbors(sulfur)
            if len(sulfur_heavy) != 2:
                continue
            external = [atom.GetIdx() for atom in sulfur_heavy if atom.GetIdx() != carbon_idx]
            if len(external) == 1:
                candidates.append((sulfur_idx, external[0], {sulfur_idx, carbon_idx}))

    elif target_type == "LYS":
        query = Chem.MolFromSmarts("[N:1]-[C:2]-[C:3]")
        for nitrogen_idx, carbon_1_idx, carbon_2_idx in mol.GetSubstructMatches(query, uniquify=True):
            carbon_1 = mol.GetAtomWithIdx(carbon_1_idx)
            carbon_2 = mol.GetAtomWithIdx(carbon_2_idx)
            nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
            if {atom.GetIdx() for atom in heavy_neighbors(carbon_1)} != {nitrogen_idx, carbon_2_idx}:
                continue
            if {atom.GetIdx() for atom in heavy_neighbors(carbon_2)} != {carbon_1_idx}:
                continue
            external = [
                atom.GetIdx()
                for atom in heavy_neighbors(nitrogen)
                if atom.GetIdx() != carbon_1_idx
            ]
            if len(external) == 1:
                candidates.append(
                    (nitrogen_idx, external[0], {nitrogen_idx, carbon_1_idx, carbon_2_idx})
                )

    elif target_type == "TYR":
        query = Chem.MolFromSmarts("[O:1]-[c:2]1[c:3][c:4][c:5][c:6][c:7]1")
        for match in mol.GetSubstructMatches(query, uniquify=True):
            oxygen_idx = match[0]
            ring_atoms = set(match[1:])
            oxygen = mol.GetAtomWithIdx(oxygen_idx)
            external = [
                atom.GetIdx()
                for atom in heavy_neighbors(oxygen)
                if atom.GetIdx() not in ring_atoms
            ]
            if len(external) == 1:
                candidates.append((oxygen_idx, external[0], {oxygen_idx} | ring_atoms))

    elif target_type == "HIS":
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) != 5:
                continue
            ring_atoms = set(ring)
            elements = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
            if elements.count("N") != 2 or elements.count("C") != 3:
                continue
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() != "N":
                    continue
                external = [
                    neighbor.GetIdx()
                    for neighbor in heavy_neighbors(atom)
                    if neighbor.GetIdx() not in ring_atoms
                ]
                for external_idx in external:
                    candidates.append((atom_idx, external_idx, ring_atoms))
    else:
        fail(f"Unsupported covalent target {target_type}.")

    return candidates


def identify_covalent_context(mol_filepath, target_type):
    """Identify the target mimic by SMARTS and validate it by graph cleavage."""

    mol = Chem.MolFromMolFile(mol_filepath, removeHs=False, sanitize=True, strictParsing=False)
    if mol is None:
        mol = Chem.MolFromMolFile(mol_filepath, removeHs=False, sanitize=False, strictParsing=False)
    if mol is None:
        fail(f"RDKit could not read optimized mol file for covalent detection: {mol_filepath}")
    if mol.GetNumAtoms() < POLY_O_ATOM_ID:
        fail(f"Optimized mol has only {mol.GetNumAtoms()} atoms; capped backbone IDs are unavailable.")

    ca_idx = POLY_CA_ATOM_ID - 1
    if mol.GetAtomWithIdx(ca_idx).GetSymbol() != "C":
        fail(f"Expected polymer CA at atom {POLY_CA_ATOM_ID}, but found {mol.GetAtomWithIdx(ca_idx).GetSymbol()}.")

    valid = []
    for target_idx, connect_idx, target_heavy_atoms in covalent_template_candidates(mol, target_type):
        if mol.GetBondBetweenAtoms(target_idx, connect_idx) is None:
            continue
        target_component = component_without_bond(
            mol, target_idx, target_idx, connect_idx
        )
        ncaa_component = component_without_bond(
            mol, connect_idx, target_idx, connect_idx
        )
        if ca_idx not in ncaa_component or ca_idx in target_component:
            continue
        component_heavy_atoms = {
            idx
            for idx in target_component
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        }
        if component_heavy_atoms != set(target_heavy_atoms):
            continue
        if target_component & ncaa_component:
            continue
        valid.append((target_idx, connect_idx, target_component))

    unique = {}
    for target_idx, connect_idx, target_component in valid:
        unique[(target_idx, connect_idx, tuple(sorted(target_component)))] = (
            target_idx,
            connect_idx,
            target_component,
        )
    valid = list(unique.values())

    if not valid:
        fail(
            f"No unique {target_type} target-mimic fragment was found in {mol_filepath}. "
            "Use the complete covalent adduct (for CYS, the product must contain NCAA-S-CH3, not NCAA-Cl)."
        )
    if len(valid) != 1:
        summary = [
            {
                "target_atom_id": target_idx + 1,
                "connect_atom_id": connect_idx + 1,
                "target_component_atom_ids": [idx + 1 for idx in sorted(component)],
            }
            for target_idx, connect_idx, component in valid
        ]
        fail(f"Covalent target detection is ambiguous: {json.dumps(summary, indent=2)}")

    target_idx, connect_idx, target_component = valid[0]
    ignored_target_ids = sorted(idx + 1 for idx in target_component if idx != target_idx)
    ignored_ids = sorted(set(CAP_IGNORE_ATOM_IDS) | set(ignored_target_ids))
    kept_params_atom_ids = [
        idx + 1
        for idx in range(mol.GetNumAtoms())
        if idx + 1 not in ignored_ids
        and idx + 1 not in (POLY_LOWER_ATOM_ID, POLY_UPPER_ATOM_ID)
    ]
    context = {
        "target_type": target_type,
        "mol_filepath": mol_filepath,
        "atom_count": mol.GetNumAtoms(),
        "target_atom_id": target_idx + 1,
        "connect_atom_id": connect_idx + 1,
        "target_component_atom_ids": [idx + 1 for idx in sorted(target_component)],
        "ignored_target_atom_ids": ignored_target_ids,
        "poly_ignore_atom_ids": ignored_ids,
        "kept_params_atom_ids": kept_params_atom_ids,
    }
    print(
        f"Detected {target_type} covalent mimic: target atom {target_idx + 1} "
        f"({mol.GetAtomWithIdx(target_idx).GetSymbol()}), NCAA connection atom "
        f"{connect_idx + 1} ({mol.GetAtomWithIdx(connect_idx).GetSymbol()}); "
        f"ignoring target atoms {ignored_target_ids}."
    )
    return context
    
def process_log_files(input_file, output_dir, covalent_target=None, chirality="auto"):
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    
    log_files = [f for f in os.listdir(output_dir) if f.endswith('.log')]
    chirality_dict = {}
    covalent_contexts = {}

    for log_file in log_files:
        base_name = os.path.splitext(log_file)[0]
        idx = int(base_name.split('_')[-1]) - 1  # assuming the log files are named like 'name_1.log', 'name_2.log', etc.
        smiles = smiles_list[idx]

        #判断结构优化是否正常结束，以防止因非正常原因停止的Gaussian结构优化所输出的log文件被应用于后续的处理过程中
        if not is_gaussian_job_completed(os.path.join(output_dir, log_file)):
            print(f"Gaussian job for {log_file} is not yet completed. Skipping for now.")
            continue

        #判断氨基酸为L型还是D型
        if chirality in ("L", "D"):
            chirality_dict[log_file] = chirality
        elif '[C@@H]' in smiles:
            chirality_dict[log_file] = 'L'
        elif '[C@H]' in smiles:
            chirality_dict[log_file] = 'D'
        else:
            print(f"Cannot determine chirality for {log_file} from SMILES: {smiles}")
            continue
        
        mol_output_dir = 'mol'
        if not os.path.exists(mol_output_dir):
            os.makedirs(mol_output_dir)
        
        #将log文件转化为mol文件。
        log_filepath = os.path.join(output_dir, log_file)
        mol_filepath = os.path.join(mol_output_dir, f'{base_name}_opt.mol')
        command = f'obabel -i log {log_filepath} -o mol -O {mol_filepath}'
        run_command(command, f"Convert Gaussian log to mol for {base_name}")
        if not os.path.exists(mol_filepath) or os.path.getsize(mol_filepath) == 0:
            fail(f"Open Babel did not create a valid mol file: {mol_filepath}")
        
        with open(mol_filepath, 'r') as mol_file:
            mol_lines = mol_file.readlines()

        covalent_context = None
        if covalent_target:
            covalent_context = identify_covalent_context(mol_filepath, covalent_target)
            covalent_contexts[base_name] = covalent_context
            context_path = os.path.join(mol_output_dir, f'{base_name}_covalent.json')
            with open(context_path, 'w') as context_file:
                json.dump(covalent_context, context_file, indent=2)
            print(f"Saved covalent atom mapping to {context_path}")
        
        # 删除最后一行"M  END"
        if mol_lines[-1].strip() == "M  END":
            mol_lines = mol_lines[:-1]

        poly_ignore_ids = list(CAP_IGNORE_ATOM_IDS)
        if covalent_context:
            poly_ignore_ids = covalent_context['poly_ignore_atom_ids']
        poly_ignore_text = " ".join(str(atom_id) for atom_id in poly_ignore_ids)

        # 根据氨基酸类型添加原子指认信息
        if chirality_dict[log_file] == 'L':
            mol_lines.extend([
                "M  ROOT 13\n",
                "M  POLY_N_BB 13\n",
                "M  POLY_CA_BB 14\n",
                "M  POLY_C_BB 15\n",
                "M  POLY_O_BB 16\n",
                f"M  POLY_IGNORE {poly_ignore_text}\n",
                "M  POLY_UPPER 7\n",
                "M  POLY_LOWER 1\n",
                "M  POLY_PROPERTIES PROTEIN L_AA ALPHA_AA\n",
                "M  END\n"
            ])
        elif chirality_dict[log_file] == 'D':
            mol_lines.extend([
                "M  ROOT 13\n",
                "M  POLY_N_BB 13\n",
                "M  POLY_CA_BB 14\n",
                "M  POLY_C_BB 15\n",
                "M  POLY_O_BB 16\n",
                f"M  POLY_IGNORE {poly_ignore_text}\n",
                "M  POLY_UPPER 7\n",
                "M  POLY_LOWER 1\n",
                "M  POLY_PROPERTIES PROTEIN D_AA ALPHA_AA\n",
                "M  END\n"
            ])
        
        with open(mol_filepath, 'w') as mol_file:
            mol_file.writelines(mol_lines)

        print(f"Processed and saved {mol_filepath}")

        # 读取并打印mol文件的内容
        with open(mol_filepath, 'r') as mol_print:
            mol_content = mol_print.read()
            print(mol_content)

    return covalent_contexts

def molfile_to_params(mol_filepath, name):
    # Rosetta names the output params from -n/--name, not from the input mol filename.
    params_filename = f"{name}.params"
    params_filepath = os.path.join(os.getcwd(), params_filename)

    # 检查参数文件是否已经存在，如果存在则skip
    if os.path.exists(params_filepath):
        print(f"Params file {params_filepath} already exists. Skipping conversion.")
        return
    
    # 通过命令行调用molfile_to_params_polymer.py脚本，执行参数化
    # 实际使用中需要根据当前操作环境下的脚本路径对以下命令进行修改
    if shutil.which(rosetta_python) is None:
        fail(f"Cannot find Rosetta params Python interpreter '{rosetta_python}'. Install python2 or set ROSETTA_PYTHON to the correct interpreter path.")

    rosetta_script = f"{rosetta_path}/main/demos/public/using_ncaas_protein_peptide_interface_design/HowToMakeResidueTypeParamFiles/scripts/molfile_to_params_polymer.py"
    if not os.path.exists(rosetta_script):
        fail(f"Required Rosetta params script is missing: {rosetta_script}")

    command = f"{rosetta_python} {rosetta_script} -n {name} --polymer {mol_filepath}"
    run_command(command, f"Generate Rosetta params for {mol_filepath}", capture_output=True)
    if not os.path.exists(params_filepath):
        fail(f"Rosetta params command finished but expected file is missing: {params_filepath}")

    # 打印成功信息
    print(f"Converted {mol_filepath} to {params_filepath}")
    
def molfile_to_params_temps(mol_filepath, name):
    # 创建temps.params路径
    params_filename = f"{name}_temps.params"
    params_filepath = os.path.join(os.getcwd(), params_filename)

    # 检查参数文件是否已经存在
    if os.path.exists(params_filepath):
        print(f"Params file {params_filepath} already exists. Skipping conversion.")
        return
    
    # 使用molfile_to_params_polymer_modify.py脚本进行参数化
    if shutil.which(rosetta_python) is None:
        fail(f"Cannot find Rosetta params Python interpreter '{rosetta_python}'. Install python2 or set ROSETTA_PYTHON to the correct interpreter path.")

    rosetta_script = f"{rosetta_path}/main/demos/public/using_ncaas_protein_peptide_interface_design/HowToMakeResidueTypeParamFiles/scripts/molfile_to_params_polymer_modify.py"
    if not os.path.exists(rosetta_script):
        fail(f"Required no-reorder Rosetta params script is missing: {rosetta_script}")

    command = f"{rosetta_python} {rosetta_script} -n {name}_temps --no_reorder --polymer {mol_filepath}"
    run_command(command, f"Generate no-reorder temp Rosetta params for {mol_filepath}", capture_output=True)
    if not os.path.exists(params_filepath):
        fail(f"Rosetta temp params command finished but expected file is missing: {params_filepath}")

    # 打印运行成功信息
    print(f"Converted {mol_filepath} to {params_filepath}")
    
# 创建resp拟合的gjf文件并提交，运行完毕后使用ff14SB力场将其转化为mol2文件
def generate_opt(geom_log_path, resp):
    resp_gjf = f'{resp}.gjf'
    resp_log = f'{resp}.log'
    resp_mol2 = f'{resp}.mol2'
    res = os.path.splitext(os.path.basename(resp))[0]

    if is_resp_gaussian_job_completed(resp_log):
        if os.path.exists(resp_mol2) and os.path.getsize(resp_mol2) > 0:
            print(f"Found completed RESP Gaussian log {resp_log} and existing RESP mol2 {resp_mol2}; skipping RESP calculation.")
            return
        print(f"Found completed RESP Gaussian log {resp_log}; skipping RESP Gaussian calculation and regenerating mol2.")
    else:
        if os.path.exists(resp_log):
            print(f"Existing log {resp_log} is not a completed RESP log; it will be overwritten.")
        run_command(
            f'antechamber -i {geom_log_path} -fi gout -o {resp_gjf} -fo gcrt -gk "# HF/6-31G*  SCF=Tight  Pop=MK  iop(6/33=2,  6/41=10, 6/42=15)"',
            f"Generate RESP Gaussian input for {res}"
        )
        set_gaussian_resources(resp_gjf)
        print(f'\nGaussian RESP-calculation input file ({resp_gjf}) for {res} has already been generated by antechamber!!\n')

        run_command(
            gaussian_command(resp_gjf, resp_log),
            f"Run RESP Gaussian for {res}"
        )
        if not is_resp_gaussian_job_completed(resp_log):
            fail(f"RESP Gaussian job did not finish normally or does not look like a RESP log: {resp_log}")

    run_command(
        f'antechamber -i {resp_log} -fi gout -o {resp_mol2} -fo mol2 -at amber -pf y -c resp',
        f"Generate RESP mol2 for {res}"
    )
    if not os.path.exists(resp_mol2):
        fail(f"RESP mol2 was not generated: {resp_mol2}")
    print(f'\nThe optimized structure with RESP charge has been output to {resp_mol2} and needs to be further processed!!\n')

def atom_type_adjust(resp_folder, covalent_context=None):
    # 遍历RESP文件夹中的mol2文件
    for mol2_file in os.listdir(resp_folder):
        if mol2_file.endswith('.mol2'):
            # 提取文件名前三个字符作为res对象
            res = mol2_file[:3]

            # 查找对应的params文件
            params_file = f'{res}_temps.params'
            if not os.path.exists(params_file):
                print(f"Warning: {params_file} not found for {mol2_file}")
                continue
            
            # 读取params文件中ATOM开头行，将其全部保存进一个列表
            params_atom_lines = []
            with open(params_file, 'r') as f_params:
                for line in f_params:
                    if line.startswith('ATOM'):
                        params_atom_lines.append(line)
            
            # 读取mol2文件内容
            with open(os.path.join(resp_folder, mol2_file), 'r') as f_mol2:
                mol2_lines = f_mol2.readlines()
            
            start_idx = -1
            end_idx = -1
            
            # 找到@<TRIPOS>ATOM和@<TRIPOS>BOND之间的行的索引范围
            try:
                start_idx = mol2_lines.index('@<TRIPOS>ATOM\n') + 1
                end_idx = mol2_lines.index('@<TRIPOS>BOND\n')
            except ValueError:
                print(f"Error: Unable to find '@<TRIPOS>ATOM' or '@<TRIPOS>BOND' in {mol2_file}")
                continue
    
            # 检索这些行中第一列>=13的行，并将它们保存进另一个列表
            mol2_atom_lines_to_replace = []
            for i in range(start_idx, end_idx):
                parts = mol2_lines[i].split()
                if len(parts) >= 1:
                    try:
                        atom_index = int(parts[0])
                    except ValueError:
                        continue  # 如果无法转换为整数，跳过该行
            
                    include_atom = atom_index >= 13
                    if covalent_context:
                        include_atom = atom_index in set(covalent_context['kept_params_atom_ids'])
                    if include_atom:
                        mol2_atom_lines_to_replace.append((i, mol2_lines[i]))

            if len(mol2_atom_lines_to_replace) != len(params_atom_lines):
                fail(
                    f"Cannot map RESP atoms to params names for {mol2_file}: "
                    f"{len(mol2_atom_lines_to_replace)} retained mol2 atoms versus "
                    f"{len(params_atom_lines)} params atoms."
                )
    
            # 将params_atom_lines中的[5:9]部分覆盖mol2_atom_lines_to_replace的[7:11]部分。即使用params的原子名称来替代mol2文件中的原子名称
            for j, (i, line) in enumerate(mol2_atom_lines_to_replace):
                parts = list(line)
                if len(parts) >= 11 and j < len(params_atom_lines):
                    new_value = params_atom_lines[j][5:9]
                    parts[7:11] = new_value
                    mol2_lines[i] = ''.join(parts)
            
            # 写入更新后的mol2文件
            output_file_path = os.path.join(resp_folder, mol2_file)
            with open(output_file_path, 'w') as f_mol2:
                f_mol2.writelines(mol2_lines)
                
                    # 读取更新后的mol2文件内容
            with open(output_file_path, 'r') as f_updated_mol2:
                updated_mol2_lines = f_updated_mol2.readlines()
    #        print('\n\n\n\n\n\n\n',updated_mol2_lines)
            
            # 查找@<TRIPOS>ATOM和@<TRIPOS>BOND之间的行的索引范围
            try:
                start_idx = updated_mol2_lines.index('@<TRIPOS>ATOM\n') + 1
                end_idx = updated_mol2_lines.index('@<TRIPOS>BOND\n')
            except ValueError:
                print(f"Error: Unable to find '@<TRIPOS>ATOM' or '@<TRIPOS>BOND' in updated {mol2_file}")
                continue
            
            # 检查并调整列表中[7:8]为数字的元素，将数字移至字母后面，使原子名称符合gromacs字母在前数字在后的规范
            adjusted_lines = []
            for i in range(start_idx, end_idx):
                line = updated_mol2_lines[i]
                parts = list(line)
                if len(parts) >= 8 and parts[7].isdigit():
                    if len(parts) >= 11 and parts[9] != ' ' and parts[10] != ' ':
                        parts[11] = parts[7]
    #                    print(parts[11])
                    if len(parts) >= 10 and parts[9] != ' ':
                        parts[10] = parts[7]
    #                    print(parts[10])
                    if len(parts) >= 9 and parts[9] == ' ':
                        parts[9] = parts[7]
    #                    print(parts[9])
                    parts[7:8] = ' '
                    adjusted_lines.append(''.join(parts))
                else:
                    adjusted_lines.append(line)
    #                print('no adjust')
    #        print(updated_mol2_lines)
    #        print('\n\n\n\n\n\n\n',adjusted_lines)
            for i in range(start_idx, end_idx):
                mol2_lines[i] = adjusted_lines[i - start_idx]
                
            # 写入更新后的mol2文件
            with open(output_file_path, 'w') as f_final_mol2:
                f_final_mol2.writelines(mol2_lines)
            
            print(f"Processed {mol2_file} successfully.")

            # 读取并打印mol2文件的内容
            with open('RESP/'+mol2_file, 'r') as mol2_print:
                mol2_content = mol2_print.read()
                print(mol2_content)

def calculate_capcharge(resp_folder):

    for mol2_file in os.listdir(resp_folder):
        if mol2_file.endswith('.mol2'):
            # 提取文件名前三个字符作为res对象
            res = mol2_file[:3]
            
            # 读取mol2文件内容
            with open(os.path.join(resp_folder, mol2_file), 'r+') as f_mol2:
                mol2 = f_mol2.readlines()
                
                # 定位ATOM所在行
                end = mol2.index('@<TRIPOS>BOND\n') 
                start = mol2.index('@<TRIPOS>ATOM\n')

                ace_cap, nme_cap = 0, 0

                # 计算ACE封端电荷
                for i in range(1 + start, 7 + start):
                    lis = list(filter(None, mol2[i].replace('\n', '').split(' ')))
                    ace_cap += eval(lis[-1])
                
                # 计算NME封端电荷
                for j in range(7 + start, 13 + start):
                    lis = list(filter(None, mol2[j].replace('\n', '').split(' ')))
                    nme_cap += eval(lis[-1])

                # 四舍五入电荷值，以保留六位小数
                ace_cap, nme_cap = round(ace_cap, 6), round(nme_cap, 6)
                
                # 写入封端电荷信息
                charge_info = open(f'{res}_cap.charge', 'w')
                charge_info.write(f'ace_cap: {ace_cap}\n')
                print(f'Sum charge of ACE: {ace_cap}')
                charge_info.write(f'nme_cap: {nme_cap}\n\n')
                print(f'Sum charge of NME: {nme_cap}')
                charge_info.close()
                
                # 删除生成的电荷信息文件
                os.remove(f'{res}_cap.charge')
                
                # N端与ACE封端电荷相加
                N_ncaa_line = mol2[start + 13]
                N_ncaa = list(filter(None, N_ncaa_line.replace('\n', '').split(' ')))
                if N_ncaa[1] == 'N':
                    print(f'Detect N-termini, with original charge {N_ncaa[-1]}')
                    new_N_charge = round(eval(N_ncaa_line[-10:-1]) + ace_cap, 6)
                    new_N_charge_str = f'{new_N_charge:9.6f}'  # 确保电荷值的格式化
                    mol2[start + 13] = N_ncaa_line.replace(N_ncaa_line[-10:-1], str(new_N_charge), 1)
                    print(f'Update N-termini charge with new value {new_N_charge}')
                else:
                    raise Exception("Please check your PDB input, ensure N-CA-C-O order.")

                # C端与NME封端电荷相加
                C_ncaa_line = mol2[start + 15]
                C_ncaa = list(filter(None, C_ncaa_line.replace('\n', '').split(' ')))
                if C_ncaa[1] == 'C':
                    print(f'Detect C-termini, with original charge {C_ncaa[-1]}')
                    new_C_charge = round(eval(C_ncaa_line[-10:-1]) + nme_cap, 6)
                    new_C_charge_str = f'{new_C_charge:9.6f}'  # 确保电荷值的格式化
                    mol2[start + 15] = C_ncaa_line.replace(C_ncaa_line[-9:-1], str(new_C_charge), 1)
                    print(f'Update C-termini charge with new value {new_C_charge}')
                else:
                    raise Exception("Please check your PDB input, ensure N-CA-C-O order.")

# 计算params二面角，用于修复params原子树构建时可能出现的异常几何
def calculate_dihedral_params(coords1, coords2, coords3, coords4):
    def vector_subtract(a, b):
        return [a[i] - b[i] for i in range(3)]

    def vector_dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    def vector_cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]

    def vector_magnitude(v):
        return math.sqrt(sum(x**2 for x in v))

    def vector_normalize(v):
        mag = vector_magnitude(v)
        if mag == 0:
            fail("Cannot calculate dihedral from zero-length vector.")
        return [x / mag for x in v]

    p = coords1
    q = coords2
    r = coords3
    s = coords4

    pq = vector_subtract(q, p)
    qr = vector_subtract(r, q)
    rs = vector_subtract(s, r)

    pq_cross_qr = vector_cross(pq, qr)
    qr_cross_rs = vector_cross(qr, rs)

    pq_cross_qr_mag = vector_magnitude(pq_cross_qr)
    qr_cross_rs_mag = vector_magnitude(qr_cross_rs)
    if pq_cross_qr_mag == 0 or qr_cross_rs_mag == 0:
        fail("Cannot calculate dihedral from collinear atoms.")

    pq_cross_qr_dot_qr_cross_rs = vector_dot(pq_cross_qr, qr_cross_rs)
    cos_dihedral = pq_cross_qr_dot_qr_cross_rs / (pq_cross_qr_mag * qr_cross_rs_mag)

    qr_normalized = vector_normalize(qr)
    pq_cross_qr_cross_qr = vector_cross(pq_cross_qr, qr_normalized)
    sin_dihedral = vector_dot(pq_cross_qr_cross_qr, qr_cross_rs) / (pq_cross_qr_mag * qr_cross_rs_mag)

    return math.degrees(math.atan2(sin_dihedral, cos_dihedral))

# 计算params角，用于修复params原子树构建时可能出现的异常几何
def calculate_angle_params(coords1, coords2, coords3):
    a = coords1
    b = coords2
    c = coords3

    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]

    ba_dot_bc = sum([ba[i] * bc[i] for i in range(3)])
    ba_mag = math.sqrt(sum([ba[i]**2 for i in range(3)]))
    bc_mag = math.sqrt(sum([bc[i]**2 for i in range(3)]))
    if ba_mag == 0 or bc_mag == 0:
        fail("Cannot calculate angle from zero-length vector.")

    cos_angle = ba_dot_bc / (ba_mag * bc_mag)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))

def process_params_file(n_value):
    filename = f'{n_value}.params'
    
    # 直接从 RESP mol2 的 ATOM 块读取坐标。这里必须保留 Rosetta 原子名，
    # 不能使用 AmberTools-safe mol2，否则 N/CA/C/O/CB 等名称会被改掉。
    mol2_file = f'RESP/{n_value}_1.mol2'
    atom_coords = {}
    in_atom_block = False
    with open(mol2_file, 'r') as file:
        for line in file:
            if line.startswith('@<TRIPOS>ATOM'):
                in_atom_block = True
                continue
            if line.startswith('@<TRIPOS>') and not line.startswith('@<TRIPOS>ATOM'):
                in_atom_block = False
                continue
            if in_atom_block and line.strip():
                parts = line.split()
                if len(parts) < 5:
                    fail(f"Unexpected mol2 atom line while reading coordinates: {line.rstrip()}")
                atom_name = parts[1]
                coords = [float(parts[2]), float(parts[3]), float(parts[4])]
                atom_coords[atom_name] = coords

    required_atoms = ['O', 'C', 'CA', 'N1', 'CB', 'N']
    missing_atoms = [atom for atom in required_atoms if atom not in atom_coords]
    if missing_atoms:
        fail(f"Cannot repair params ICOOR entries because RESP mol2 is missing atoms {missing_atoms}: {mol2_file}")

    # 计算所需值
    value_1 = calculate_dihedral_params(atom_coords['O'], atom_coords['C'], atom_coords['CA'], atom_coords['N1'])
    value_2 = 180 - calculate_angle_params(atom_coords['O'], atom_coords['C'], atom_coords['CA'])
    value_3 = math.sqrt(sum([(atom_coords['O'][i] - atom_coords['C'][i])**2 for i in range(3)]))
    
    value_4 = calculate_dihedral_params(atom_coords['CB'], atom_coords['CA'], atom_coords['N'], atom_coords['C'])
    value_5 = 180 - calculate_angle_params(atom_coords['CB'], atom_coords['CA'], atom_coords['N'])
    value_6 = math.sqrt(sum([(atom_coords['CB'][i] - atom_coords['CA'][i])**2 for i in range(3)]))
    
    # 读取params文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # 查找并处理以“ICOOR_INTERNAL    O ”开头的行
    for i, line in enumerate(lines):
        if line.startswith('ICOOR_INTERNAL    O '):
            parts = line.split()
            if len(parts) == 8:
                dihedral_angle = float(parts[2])
                bond_angle = float(parts[3])
                
                if not (179 <= dihedral_angle <= 181 or -181 <= dihedral_angle <= -179) or not (59 <= bond_angle <= 61):
                    new_line = f'ICOOR_INTERNAL    O    {value_1:.6f}   {value_2:.6f}    {value_3:.6f}   C     CA  UPPER\n'
                    lines[i] = new_line
                    break
                    
    # 查找并处理以“ICOOR_INTERNAL    CB”开头的行
    for i, line in enumerate(lines):
        if line.startswith('ICOOR_INTERNAL    CB'):
            parts = line.split()
            if len(parts) == 8:
                dihedral_angle = float(parts[2])
                bond_angle = float(parts[3])
                
                if not (-123 <= dihedral_angle <= -121 or 121 <= dihedral_angle <= 123) or not (69 <= bond_angle <= 71):
                    new_line = f'ICOOR_INTERNAL    CB  -{value_4:.6f}   {value_5:.6f}    {value_6:.6f}   CA    N     C \n'
                    lines[i] = new_line
                    break
    
    # 将修改后的内容写回文件
    with open(filename, 'w') as file:
        file.writelines(lines)
    
    # 打印params文件的内容
    with open(filename, 'r') as params_print:
        params_content = params_print.read()
        print(params_content)

def read_mol2_file(mol2_filepath, included_atom_ids=None):
    atom_lines = []
    with open(mol2_filepath, 'r') as f:
        lines = f.readlines()
    
    atom_started = False
    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            atom_started = True
            continue
        if line.startswith('@<TRIPOS>BOND'):
            atom_started = False
            continue
        if atom_started and line.strip():  # only capture non-empty lines between ATOM and BOND sections
            atom_lines.append(line.strip())
    
    if included_atom_ids is None:
        return atom_lines[12:]  # Skip the first 12 lines
    included = set(included_atom_ids)
    selected = []
    for line in atom_lines:
        parts = line.split()
        if parts and int(parts[0]) in included:
            selected.append(line)
    return selected

def read_params_file(params_filepath):
    atom_lines = []
    with open(params_filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('ATOM'):
            atom_lines.append(line.strip())
    
    return atom_lines

def adjust_charges_to_integer(charges_list,system_charge):
    #计算当前净电荷距离整数电荷的差值
    rounded_charges = [round(charge, 2) for charge in charges_list]

    charge_diff = system_charge - sum(rounded_charges)
    
    if charge_diff == 0:
        return rounded_charges

    # 按照电荷绝对值从大到小排序
    sorted_indices = sorted(range(len(charges_list)), key=lambda i: abs(charges_list[i]), reverse=True)

    # 按电荷从大到小依次进行±0.01的调整，从而在对体系电荷产生最小影响的情况下使电荷值为整数
    for _ in range(abs(int(charge_diff * 100))):  # 需要的调整次数
        for i in sorted_indices:
            if charge_diff > 0:
                rounded_charges[i] += 0.01
                charge_diff -= 0.01
            elif charge_diff < 0:
                rounded_charges[i] -= 0.01
                charge_diff += 0.01
            if round(charge_diff, 2) == 0:
                break
    
    return rounded_charges

def modify_params_file(params_filepath, rounded_charges):
    with open(params_filepath, 'r') as f:
        lines = f.readlines()

    atom_lines = []
    atom_indices = []

    #读取params文件的ATOM行
    for idx, line in enumerate(lines):
        if line.startswith('ATOM'):
            atom_lines.append(line.strip())
            atom_indices.append(idx)

    #检查原子数和电荷列表数是否匹配
    if len(atom_lines) != len(rounded_charges):
        print(f"ATOM lines count: {len(atom_lines)}")
        print(f"Rounded charges count: {len(rounded_charges)}")
        raise ValueError('params 文件中的 ATOM 行数与电荷列表长度不匹配。')

    for i in range(len(atom_lines)):
        original_line = atom_lines[i]
        # 提取新的电荷值
        charge_part = original_line[-5:]  # 提取最后的电荷部分
        new_charge = f"{rounded_charges[i]:.2f}"
        
        # 检查正数电荷值并在前面加空格以保证格式统一
        if float(new_charge) > 0:
            new_charge = f" {new_charge}"
        
        new_line = original_line[:-5] + new_charge + '\n'  # 应用新的电荷值组装新的行内容
        lines[atom_indices[i]] = new_line  # 更新原始文件中的ATOM行

    # 将修改后的 ATOM 行写回文件
    with open(params_filepath, 'w') as f:
        f.writelines(lines)

def update_params_file_with_temps(params_filepath, temps_filepath):
    # 读取params文件中的所有行
    with open(params_filepath, 'r') as f:
        params_lines = f.readlines()

    # 读取temps文件中的所有以ATOM开头的行
    temps_atoms = []
    with open(temps_filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                temps_atoms.append(line.strip())

    # 遍历params文件中的所有以ATOM开头的行，进行替换
    updated_params_lines = []
    for line in params_lines:
        if line.startswith('ATOM'):
            atom_id = line[:18]
            for temps_line in temps_atoms:
                if temps_line[:18] == atom_id:
                    line = temps_line + '\n'
                    break
        updated_params_lines.append(line)

    # 将更新后的行写回params文件
    with open(params_filepath, 'w') as f:
        f.writelines(updated_params_lines)

    # 读取并打印params文件的内容
    with open(params_filepath, 'r') as params_new_print:
        params_new_content = params_new_print.read()
        print(params_new_content)


def resolve_covalent_atom_names(context, temps_filepath):
    atom_names = read_params_atom_names(temps_filepath)
    kept_ids = context['kept_params_atom_ids']
    if len(atom_names) != len(kept_ids):
        fail(
            f"Cannot resolve covalent atom names: {len(atom_names)} params atoms versus "
            f"{len(kept_ids)} retained mol atoms."
        )
    atom_name_by_id = {
        str(atom_id): atom_name for atom_id, atom_name in zip(kept_ids, atom_names)
    }
    target_id = str(context['target_atom_id'])
    connect_id = str(context['connect_atom_id'])
    if target_id not in atom_name_by_id or connect_id not in atom_name_by_id:
        fail("Covalent target or connection atom was removed before params postprocessing.")
    context['atom_name_by_id'] = atom_name_by_id
    context['target_atom_name'] = atom_name_by_id[target_id]
    context['connect_atom_name'] = atom_name_by_id[connect_id]
    return context


def is_amide_bond(mol, atom_a_idx, atom_b_idx):
    """Return True for a C(=O)-N bond, irrespective of atom order."""

    atom_a = mol.GetAtomWithIdx(atom_a_idx)
    atom_b = mol.GetAtomWithIdx(atom_b_idx)
    if {atom_a.GetAtomicNum(), atom_b.GetAtomicNum()} != {6, 7}:
        return False
    carbon = atom_a if atom_a.GetAtomicNum() == 6 else atom_b
    nitrogen = atom_b if carbon is atom_a else atom_a
    for neighbor in carbon.GetNeighbors():
        if neighbor.GetIdx() == nitrogen.GetIdx() or neighbor.GetAtomicNum() != 8:
            continue
        bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), neighbor.GetIdx())
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False


def canonical_covalent_chis(context):
    """Build sidechain CHIs in CA-to-CONN3 order from the optimized molecule.

    molfile_to_params_polymer can emit chemically valid rotatable bonds in an
    order that does not follow the sidechain, and may include an amide C-N
    bond.  Covalent NCAA rotamers need a stable proximal-to-distal ordering.
    Ring, multiple, aromatic, and amide bonds are deliberately excluded.
    """

    mol_filepath = context['mol_filepath']
    mol = Chem.MolFromMolFile(
        mol_filepath, removeHs=False, sanitize=True, strictParsing=False
    )
    if mol is None:
        mol = Chem.MolFromMolFile(
            mol_filepath, removeHs=False, sanitize=False, strictParsing=False
        )
    if mol is None:
        fail(f"RDKit could not reload optimized molecule for CHI generation: {mol_filepath}")

    ca_idx = POLY_CA_ATOM_ID - 1
    n_idx = POLY_N_ATOM_ID - 1
    connect_idx = context['connect_atom_id'] - 1
    path = list(Chem.GetShortestPath(mol, ca_idx, connect_idx))
    if not path or path[0] != ca_idx or path[-1] != connect_idx:
        fail(
            f"Cannot trace the covalent sidechain from CA atom {POLY_CA_ATOM_ID} "
            f"to connection atom {context['connect_atom_id']}."
        )

    atom_name_by_id = context['atom_name_by_id']

    def atom_name(atom_idx):
        atom_id = str(atom_idx + 1)
        if atom_id not in atom_name_by_id:
            fail(
                f"Sidechain CHI path uses atom {atom_id}, which is absent from the "
                "retained params atom mapping."
            )
        return atom_name_by_id[atom_id]

    chis = []
    skipped = []
    for path_index in range(len(path) - 1):
        atom_a_idx = path[path_index]
        atom_b_idx = path[path_index + 1]
        bond = mol.GetBondBetweenAtoms(atom_a_idx, atom_b_idx)
        if bond is None:
            fail(f"Missing bond on covalent CHI path: {atom_a_idx + 1}-{atom_b_idx + 1}")

        reason = None
        if bond.GetBondTypeAsDouble() != 1.0 or bond.GetIsAromatic():
            reason = 'non-single or aromatic bond'
        elif bond.IsInRing():
            reason = 'ring bond'
        elif is_amide_bond(mol, atom_a_idx, atom_b_idx):
            reason = 'amide C-N bond'
        if reason:
            skipped.append(
                {
                    'atom_ids': [atom_a_idx + 1, atom_b_idx + 1],
                    'atom_names': [atom_name(atom_a_idx), atom_name(atom_b_idx)],
                    'reason': reason,
                }
            )
            continue

        atom_1_idx = n_idx if path_index == 0 else path[path_index - 1]
        atom_4 = (
            atom_name(path[path_index + 2])
            if path_index + 2 < len(path)
            else 'V1'
        )
        chis.append(
            [
                atom_name(atom_1_idx),
                atom_name(atom_a_idx),
                atom_name(atom_b_idx),
                atom_4,
            ]
        )

    if not chis:
        fail("No rotatable sidechain CHIs remain after covalent CHI normalization.")
    context['covalent_sidechain_path_atom_ids'] = [idx + 1 for idx in path]
    context['skipped_covalent_path_bonds'] = skipped
    return chis


def replace_active_chis(lines, chis):
    """Replace active CHI records while preserving commented audit records."""

    first_chi_index = next(
        (index for index, line in enumerate(lines) if line.startswith('CHI ')),
        next(
            (index for index, line in enumerate(lines) if line.startswith('NBR_ATOM')),
            len(lines),
        ),
    )
    without_active_chis = [line for line in lines if not line.startswith('CHI ')]
    removed_before_insertion = sum(
        1
        for line in lines[:first_chi_index]
        if line.startswith('CHI ')
    )
    insertion = first_chi_index - removed_before_insertion
    chi_lines = [
        f"CHI {number}  {atoms[0]:<4} {atoms[1]:<4} {atoms[2]:<4} {atoms[3]:<4}\n"
        for number, atoms in enumerate(chis, start=1)
    ]
    without_active_chis[insertion:insertion] = chi_lines
    return without_active_chis


def postprocess_covalent_params(params_filepath, temps_filepath, context):
    """Convert the retained target nucleophile atom into CONNECT/CONN3/V1."""

    context = resolve_covalent_atom_names(context, temps_filepath)
    target_name = context['target_atom_name']
    connect_name = context['connect_atom_name']

    with open(params_filepath, 'r') as params_file:
        lines = params_file.readlines()

    target_atom_line = None
    target_icoor = None
    bond_found = False
    connect_found = False
    terminal_chi_count = 0
    removed_target_chis = []
    output = []

    for line in lines:
        stripped = line.strip()
        parts = stripped.split()

        if line.startswith('ATOM') and len(parts) >= 5 and parts[1] == target_name:
            target_atom_line = line
            output.append('#' + line)
            suffix = ' '.join(parts[4:])
            output.append(f"ATOM  V1  VIRT VIRT  {suffix}\n")
            continue

        if (line.startswith('BOND ') or line.startswith('BOND_TYPE ')) and len(parts) >= 3:
            atom_pair = {parts[1], parts[2]}
            if atom_pair == {target_name, connect_name}:
                bond_found = True
                output.append('#' + line)
                output.append(f"BOND  {connect_name:<4} V1\n")
                continue

        if line.startswith('CONNECT ') and len(parts) >= 2 and parts[1] == connect_name:
            connect_found = True

        if line.startswith('ICOOR_INTERNAL') and len(parts) == 8:
            atom_name = parts[1]
            if atom_name == target_name:
                if parts[5] != connect_name:
                    fail(
                        f"Target atom {target_name} is not parented by connection atom "
                        f"{connect_name} in {params_filepath}: {stripped}"
                    )
                target_icoor = parts
                output.append('#' + line)
                output.append(
                    f"ICOOR_INTERNAL   CONN3 {float(parts[2]):11.6f} "
                    f"{float(parts[3]):11.6f} {float(parts[4]):11.6f}   "
                    f"{parts[5]:<4}  {parts[6]:<4}  {parts[7]:<4}\n"
                )
                output.append(
                    f"ICOOR_INTERNAL    V1   {0.0:11.6f} "
                    f"{float(parts[3]):11.6f} {float(parts[4]):11.6f}   "
                    f"{parts[5]:<4}  {parts[6]:<4}  CONN3\n"
                )
                continue

            replaced_parts = list(parts)
            replaced = False
            for index in range(5, 8):
                if replaced_parts[index] == target_name:
                    replaced_parts[index] = 'CONN3'
                    replaced = True
            if replaced:
                output.append(
                    f"ICOOR_INTERNAL {replaced_parts[1]:>6} "
                    f"{float(replaced_parts[2]):11.6f} {float(replaced_parts[3]):11.6f} "
                    f"{float(replaced_parts[4]):11.6f}   {replaced_parts[5]:<4}  "
                    f"{replaced_parts[6]:<4}  {replaced_parts[7]:<5}\n"
                )
                continue

        output.append(line)

    if target_atom_line is None:
        fail(f"Target atom {target_name} was not found in {params_filepath}.")
    if target_icoor is None:
        fail(f"Target atom ICOOR for {target_name} was not found in {params_filepath}.")
    if not bond_found:
        fail(f"Target bond {connect_name}-{target_name} was not found in {params_filepath}.")

    if not connect_found:
        insertion = next(
            (index + 1 for index, line in enumerate(output) if line.startswith('UPPER_CONNECT')),
            None,
        )
        if insertion is None:
            fail(f"UPPER_CONNECT was not found in {params_filepath}.")
        output.insert(insertion, f"CONNECT {connect_name}\n")

    terminal_atoms = [target_icoor[7], target_icoor[6], connect_name]
    for index, line in enumerate(output):
        parts = line.split()
        if not line.startswith('CHI ') or len(parts) < 6:
            continue
        if parts[2:5] == terminal_atoms and parts[5] == target_name:
            parts[5] = 'V1'
            output[index] = (
                f"CHI {parts[1]}  {parts[2]:<4} {parts[3]:<4} {parts[4]:<4} {parts[5]:<4}\n"
            )
            terminal_chi_count += 1
        elif target_name in parts[2:6]:
            # molfile_to_params can emit downstream CHIs through POLY_IGNORE atoms.
            # They are invalid after the target-mimic component is removed.
            removed_target_chis.append(line.strip())
            output[index] = '#' + line

    if terminal_chi_count > 1:
        fail(f"Found multiple terminal CHIs for covalent target {target_name}.")
    if terminal_chi_count == 0:
        chi_indices = [
            (index, int(line.split()[1]))
            for index, line in enumerate(output)
            if line.startswith('CHI ')
        ]
        chi_number = max((number for _, number in chi_indices), default=0) + 1
        insertion = chi_indices[-1][0] + 1 if chi_indices else next(
            (index for index, line in enumerate(output) if line.startswith('NBR_ATOM')),
            len(output),
        )
        output.insert(
            insertion,
            f"CHI {chi_number}  {terminal_atoms[0]:<4} {terminal_atoms[1]:<4} "
            f"{terminal_atoms[2]:<4} V1\n",
        )

    active_target_references = []
    for line in output:
        if line.startswith('#'):
            continue
        parts = line.split()
        if target_name in parts and parts and parts[0] in {
            'ATOM', 'BOND', 'BOND_TYPE', 'CHI', 'ICOOR_INTERNAL'
        }:
            active_target_references.append(line.rstrip())
    if active_target_references:
        fail(
            f"Covalent postprocessing left active references to {target_name}: "
            f"{active_target_references}"
        )

    generated_chis = [line.strip() for line in output if line.startswith('CHI ')]
    canonical_chis = canonical_covalent_chis(context)
    output = replace_active_chis(output, canonical_chis)
    final_chis = [line.strip() for line in output if line.startswith('CHI ')]
    context['pre_normalization_chi_definitions'] = generated_chis
    context['final_chi_definitions'] = final_chis
    context['removed_target_chi_definitions'] = removed_target_chis

    with open(params_filepath, 'w') as params_file:
        params_file.writelines(output)

    print(
        f"Converted {params_filepath} to a covalent residue: CONNECT {connect_name}, "
        f"target {target_name} -> CONN3/V1."
    )
    for skipped_bond in context['skipped_covalent_path_bonds']:
        print(
            "Skipped covalent path bond "
            f"{skipped_bond['atom_names']} from CHI generation: "
            f"{skipped_bond['reason']}."
        )
    print("[CHI REVIEW RECOMMENDED] Canonical CA-to-CONN3 CHI definitions:")
    for chi_line in final_chis:
        print(f"  {chi_line}")
    return context
      
def process_mol2_file(file_path):
    # Read the mol2 file into a list of lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Process each line according to the specified conditions
    for i in range(len(lines)):
        line = lines[i]
        if len(line) >= 52 and line[50:52] == "DU" or line[50:52] == "N3":
            if "N" in line[8:10]:
                lines[i] = line[:50] + "N " + line[52:]

    # Write the modified lines back to the original file
    with open(file_path, 'w') as f:
        f.writelines(lines)
        
def process_mol2_with_args(mol2_value):
    # Construct the file name based on the argument
    file_name = f"{mol2_value}_1.mol2"

    # Get the full file path
    folder_path = 'RESP'  # Assuming RESP folder is in the current working directory
    file_path = os.path.join(folder_path, file_name)

    # Process the mol2 file
    process_mol2_file(file_path)
    
    # 读取并打印mol2文件的内容
    with open(file_path, 'r') as mol2_new_print:
        mol2_new_content = mol2_new_print.read()
        print(mol2_new_content)

def element_from_mol2_type(atom_type):
    token = atom_type.strip()
    upper = token.upper()
    if upper.startswith('CL'):
        return 'Cl'
    if upper.startswith('BR'):
        return 'Br'
    if upper.startswith('F'):
        return 'F'
    if upper.startswith('S'):
        return 'S'
    if upper.startswith('O'):
        return 'O'
    if upper.startswith('N'):
        return 'N'
    if upper.startswith('C'):
        return 'C'
    if upper.startswith('H'):
        return 'H'
    if upper.startswith('P'):
        return 'P'
    fail(f"Cannot infer element from mol2 atom type '{atom_type}' for AmberTools-safe mol2 conversion.")

def make_amber_safe_mol2(input_mol2, output_mol2):
    with open(input_mol2, 'r') as f:
        lines = f.readlines()

    in_atom_block = False
    element_counts = {}
    new_lines = []

    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            in_atom_block = True
            new_lines.append(line)
            continue
        if line.startswith('@<TRIPOS>') and not line.startswith('@<TRIPOS>ATOM'):
            in_atom_block = False
            new_lines.append(line)
            continue

        if in_atom_block and line.strip():
            parts = line.split()
            if len(parts) < 9:
                fail(f"Unexpected mol2 atom line while creating AmberTools-safe mol2: {line.rstrip()}")
            element = element_from_mol2_type(parts[5])
            element_counts[element] = element_counts.get(element, 0) + 1
            parts[1] = f"{element}{element_counts[element]}"
            new_line = (
                f"{int(parts[0]):7d} "
                f"{parts[1]:<8s} "
                f"{float(parts[2]):10.4f} "
                f"{float(parts[3]):10.4f} "
                f"{float(parts[4]):10.4f} "
                f"{parts[5]:<8s} "
                f"{int(parts[6]):4d} "
                f"{parts[7]:<8s} "
                f"{float(parts[8]):10.6f}\n"
            )
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(output_mol2, 'w') as f:
        f.writelines(new_lines)

    print(f"Generated AmberTools-safe mol2: {output_mol2}")
    return output_mol2

def generate_top(resp_folder):

    # 遍历RESP文件夹中的mol2文件
    for mol2_file in os.listdir(resp_folder):
        if mol2_file.endswith('.mol2') and not mol2_file.endswith('_amber.mol2'):
            # 提取文件名前三个字符作为res对象
            res = mol2_file[:3]
            global res_rtp
            res_rtp=res
            mol2_path = os.path.join(resp_folder, mol2_file)
            amber_mol2_file = f"{os.path.splitext(mol2_file)[0]}_amber.mol2"
            amber_mol2_path = os.path.join(resp_folder, amber_mol2_file)
            make_amber_safe_mol2(mol2_path, amber_mol2_path)

            # 创建存放结果的文件夹
            output_folder = f'./{res}_gromacs_prm'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # 执行parmchk2命令生成.mod文件
            mod_file = f'{res}.mod'
            run_command(f'parmchk2 -i {amber_mol2_path} -f mol2 -o {mod_file}', f"Generate Amber frcmod for {res}")

            # 生成leapin文件
            leapin_filename = f'{res}_leap.in'
            with open(leapin_filename, 'w+') as leapin:
                leapin.write('source leaprc.protein.ff19SB\n')
                leapin.write(f'loadamberparams {mod_file}\n')
                leapin.write(f'mol=loadmol2 {amber_mol2_path}\n')
                leapin.write(f'check mol\n')
                leapin.write(f'saveamberparm mol {res}.prm {res}.crd\n')
                leapin.write('quit\n')

            # 调用tleap
            run_command(f'tleap -f {leapin_filename}', f"Run tleap for {res}")

            # 移动生成的文件到指定文件夹
            generated_files = [f'{res}.prm', f'{res}.crd', mod_file, leapin_filename, 'leap.log']
            for file in generated_files:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(output_folder, file))

            # 切换到输出文件夹进行ACpype操作
            os.chdir(output_folder)
            run_command(f'acpype -p {res}.prm -x {res}.crd -c user -o gmx -a amber', f"Run ACPYPE for {res}")

            # 返回到原始工作目录
            os.chdir('..')

            # 移动生成的GROMACS文件到指定文件夹
            gromacs_files = ['MOL_GMX.gro', 'MOL_GMX.top']
            for file in gromacs_files:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(output_folder, f'{res}.{file.split("_")[-1]}'))

            # 清理中间文件
            run_command(f'rm -f qout QOUT punch md.mdp esout em.mdp', "Clean AmberTools temporary files")

    print(f'\nFZ-wang reminds you: The GROMACS top files have been generated in the folder "gromacs_prm"!\n')
    

def read_params_atom_names(params_filepath):
    atom_names = []
    with open(params_filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) < 2:
                    fail(f"Unexpected ATOM line while reading params atom names: {line.rstrip()}")
                atom_names.append(parts[1])
    return atom_names

def generate_rtp(resp_file):
    res = res_rtp  # 残基的名称或标识
    ignore_atoms = ["ACE", "NME", "linker"]  # 不需要包含在 .rtp 文件中的原子名称列表
    params_file = f'{res}.params'
    if not os.path.exists(params_file):
        params_file = f'{res}_temps.params'
    if not os.path.exists(params_file):
        fail(f"Cannot restore RTP atom names because params file is missing: {res}.params or {res}_temps.params")

    params_atom_names = read_params_atom_names(params_file)
    params_name_by_top_index = {
        idx + 13: atom_name
        for idx, atom_name in enumerate(params_atom_names)
    }
    print(f"Restoring RTP atom names from {params_file}; ACPYPE atom names will only provide force-field parameters.")

    def mapped_atom_name(atom_num):
        atom_idx = int(atom_num)
        if atom_idx < 13:
            return None
        return params_name_by_top_index.get(atom_idx)

    def mapped_atom_names(atom_nums):
        names = []
        for atom_num in atom_nums:
            atom_idx = int(atom_num)
            if atom_idx < 13:
                names.append(None)
                continue
            name = params_name_by_top_index.get(atom_idx)
            if name is None:
                fail(f"Cannot map ACPYPE atom index {atom_num} back to params atom names from {params_file}")
            names.append(name)
        return names

    def skip_atom_nums(atom_nums):
        return any(int(atom_num) < 13 for atom_num in atom_nums)

    # 打开并读取 .top 文件
    with open(resp_file, 'r') as f_top:
        top = f_top.readlines()

    # 确定各个部分的起始和结束行索引
    start_atom = top.index('[ atoms ]\n') + 2
    start_bond = top.index('[ bonds ]\n') + 2
    end_atom = top.index('[ bonds ]\n') - 1
    end_bond = top.index('[ pairs ]\n') - 1
    start_angle = top.index('[ angles ]\n') + 2
    end_angle = top.index('[ dihedrals ] ; propers\n') - 1
    start_dihedral = top.index('[ dihedrals ] ; propers\n') + 3
    end_dihedral = top.index('[ dihedrals ] ; impropers\n') - 1
    start_improper = top.index('[ dihedrals ] ; impropers\n') + 3
    end_improper = top.index('[ system ]\n') - 1

    # 初始化 RTP 列表
    rtp_list = []
    rtp_list.append(f'[ {res} ]\n')  # 残基条目
    rtp_list.append(' [ atoms ]\n')  # atoms

    # 定义 include_ffparm 函数，用于排除不需要的行
    def include_ffparm(atom_nums, atom_names):
        if skip_atom_nums(atom_nums):
            return False
        for j in atom_names:
            if j in ignore_atoms:
                return False
        return True
    
    # 处理 atoms 项
    rtp_atom_num = 0
    for i in range(start_atom, end_atom):
        line = top[i]
        atom = list(filter(None, line.replace('\n', '').split()))

        atom_nums = [atom[0]]
        if skip_atom_nums(atom_nums):
            continue
        atom_names = mapped_atom_names(atom_nums)
        rtp_atom_num += 1
        atom_name, atom_type, atom_charge, atom_num = atom_names[0], atom[1], atom[6], rtp_atom_num
        
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing atom {atom_name}')
            rtp_list.append(f'    {atom_name:>4}   {atom_type:>2}    {atom_charge:>9}    {atom_num:>2}\n')

    rtp_list.append('\n [ bonds ]\n')

    # 处理 bonds 项
    for j in range(start_bond, end_bond):
        line = top[j]
        bond = list(filter(None, line.replace('\n', '').split()))

        atom_nums = bond[0:2]
        if skip_atom_nums(atom_nums):
            continue
        atom_names = mapped_atom_names(atom_nums)
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing bond {atom_names[0]}-{atom_names[1]}')
            rtp_list.append(f'    {atom_names[0]:>4}   {atom_names[1]:<4}  {bond[3]}    {bond[4]}\n')

    rtp_list.append(f'    {"-C":>4}   {"N":<4}  1.3790e-01    3.5782e+05\n')
    rtp_list.append('\n [ angles ]\n')

    # 处理 angles 项
    for k in range(start_angle, end_angle):
        line = top[k]
        angle = list(filter(None, line.replace('\n', '').split()))

        atom_nums = angle[0:3]
        if skip_atom_nums(atom_nums):
            continue
        atom_names = mapped_atom_names(atom_nums)
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing angle {atom_names[0]}-{atom_names[1]}-{atom_names[2]}')
            rtp_list.append(f'    {atom_names[0]:>4}   {atom_names[1]:>4}    {atom_names[2]:<4}  {angle[4]}   {angle[5]}\n')

    rtp_list.append('\n [ dihedrals ] ; propers\n')

    # 处理 dihedrals 项
    for l in range(start_dihedral, end_dihedral):
        line = top[l]
        dihedral = list(filter(None, line.replace('\n', '').split()))

        atom_nums = dihedral[0:4]
        if skip_atom_nums(atom_nums):
            continue
        atom_names = mapped_atom_names(atom_nums)
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing dihedrals proper {atom_names[0]}-{atom_names[1]}-{atom_names[2]}-{atom_names[3]}')
            rtp_list.append(f'    {atom_names[0]:>4}   {atom_names[1]:>4}   {atom_names[2]:>4}   {atom_names[3]:<4}  {dihedral[5]:>6}   {dihedral[6]:>8}   {dihedral[7]}\n')

    rtp_list.append('\n [ dihedrals ] ; impropers\n')

    # 处理 impropers 项
    for m in range(start_improper, end_improper):
        line = top[m]
        improper = list(filter(None, line.replace('\n', '').split()))

        atom_nums = improper[0:4]
        if skip_atom_nums(atom_nums):
            continue
        atom_names = mapped_atom_names(atom_nums)
        if include_ffparm(atom_nums, atom_names):
#           print(f'processing dihedrals impropers {atom_names[0]}-{atom_names[1]}-{atom_names[2]}-{atom_names[3]}')
            rtp_list.append(f'    {atom_names[0]:>4}   {atom_names[1]:>4}   {atom_names[2]:>4}   {atom_names[3]:<4}  {improper[5]:>6}   {improper[6]:>8}   {improper[7]}\n')

    rtp_list.append('    -C    CA     N     H  180.00   4.60240   2\n    CA    +N     C     O  180.00   4.60240   2\n')

    # 写入生成的 .rtp 文件
    with open(f'{res_rtp}.rtp', 'w') as f_rtp:
        for line in rtp_list:
            f_rtp.write(line)

    # 读取并打印rtp文件的内容
    with open(f'{res_rtp}.rtp', 'r') as rtp_print:
        rtp_content = rtp_print.read()
        print(rtp_content)

def organize_outputs(n_value, input_file):
    molecule_dir = n_value
    artifact_dir = os.path.join(molecule_dir, "intermediates")
    os.makedirs(artifact_dir, exist_ok=True)

    core_files = [
        f'{n_value}.params',
        f'{n_value}.rtp',
    ]

    dirs_to_move = [
        'pdb_files',
        'PDB_rearranged',
        'GJF',
        'mol',
        'RESP',
        f'{n_value}_gromacs_prm',
    ]

    files_to_move = [
        f'{n_value}_temps.params',
        f'{n_value}_temps_0001.pdb',
        f'{n_value}_0001.pdb',
        f'{n_value}_covalent.json',
        'molecule.chk',
        'fort.7',
    ]

    for path in core_files:
        if os.path.exists(path):
            dest = os.path.join(molecule_dir, os.path.basename(path))
            if os.path.abspath(path) != os.path.abspath(dest):
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.move(path, dest)

    for path in dirs_to_move:
        if os.path.exists(path):
            dest = os.path.join(artifact_dir, os.path.basename(path))
            if os.path.exists(dest):
                shutil.rmtree(dest, ignore_errors=True)
            shutil.move(path, dest)

    for path in files_to_move:
        if os.path.exists(path):
            dest = os.path.join(artifact_dir, os.path.basename(path))
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(path, dest)

    if input_file and os.path.exists(input_file):
        dest = os.path.join(molecule_dir, os.path.basename(input_file))
        if os.path.abspath(input_file) != os.path.abspath(dest) and not os.path.exists(dest):
            shutil.copyfile(input_file, dest)

    print(f"Organized outputs into {molecule_dir}/")
        
def main(input_file, names_list, clean, stage, covalent_target=None, input_form='auto', chirality='auto'):
    # 确保所有需要的文件夹存在或根据需要创建它们
    required_dirs = ['pdb_files', 'PDB_rearranged', 'GJF', 'mol']
 
    
    gjf_folder = 'GJF'
    resp_folder = 'RESP'
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
 
    # 读取输入的SMILES文件
    
    # 计算体系电荷
    system_charge = calculate_system_charge(input_file)
    print(f"System charge: {system_charge}")
    
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
 
    if len(names_list) != len(smiles_list):
        raise ValueError("The number of provided names does not match the number of SMILES strings.")
    if covalent_target and len(smiles_list) != 1:
        raise ValueError("Covalent parameterization currently accepts exactly one SMILES per run.")
    
    gaussian_input_dir = 'GJF'

    if stage != 'from_logs':
        # 处理SMILES
        processed_smiles_list = []
        for smiles in smiles_list:
            try:
                processed_smiles = prepare_input_smiles(smiles, input_form)
                processed_smiles_list.append(processed_smiles)
            except ValueError as e:
                print(e)
                continue
 
        # 将处理后的SMILES转换为PDB
        output_dir = 'pdb_files'
        generated_files = smiles_to_pdb(processed_smiles_list, names_list, output_dir)
 
        # 重新排序并处理PDB文件
        output_rearranged_dir = 'PDB_rearranged'
        process_files(generated_files, names_list, output_rearranged_dir)
 
        # 调整二面角
        adjusted_pdb_files = [os.path.join(output_rearranged_dir, f'{name}_{i+1}.pdb') for i, name in enumerate(names_list)]
        adjust_dihedrals_in_pdb_files(adjusted_pdb_files)
 
        # 删除PDB_rearranged文件夹中所有PDB文件的CONECT行
        for pdb_file in os.listdir(output_rearranged_dir):
            if pdb_file.endswith(".pdb"):
                pdb_filepath = os.path.join(output_rearranged_dir, pdb_file)
                remove_conect_lines_from_pdb(pdb_filepath)
 
        # 生成Gaussian输入文件。gjf_only模式只生成gjf，不提交本地Gaussian。
        submitted_jobs = generate_gaussian_input(
            adjusted_pdb_files,
            gaussian_input_dir,
            system_charge,
            submit=(stage == 'all'),
        )

        if stage == 'gjf_only':
            print("Generated Gaussian input files in GJF/. Transfer the .gjf files to HPC, run Gaussian there, then copy completed .log files back and rerun with --stage from_logs.")
            return
    
        for log_filepath, process in submitted_jobs:
            return_code = process.wait()
            if return_code != 0:
                fail(f"Gaussian optimization exited with code {return_code}: {log_filepath}")
            if not is_gaussian_job_completed(log_filepath):
                fail(f"Gaussian optimization did not terminate normally: {log_filepath}")
            print(f"Gaussian optimization completed normally: {log_filepath}")
    else:
        print("Resuming from completed Gaussian log files in GJF/.")
    
    # 所有Gaussian作业完成后，处理log文件并生成mol文件
    covalent_contexts = process_log_files(
        input_file,
        gaussian_input_dir,
        covalent_target=covalent_target,
        chirality=chirality,
    )
    
        # 获取生成的mol文件列表
    mol_files = [f for f in os.listdir('mol') if f.endswith('.mol')]
    if not mol_files:
        raise RuntimeError("No .mol files were generated from GJF/*.log. Check that completed Gaussian .log files are present and contain 'Normal termination'.")
    
    # 遍历每个mol文件，转化为params文件并存放到对应的文件夹中
    for mol_file in mol_files:
        base_name = os.path.splitext(mol_file)[0]
        name = base_name.split('_')[0]
        mol_filepath = os.path.join('mol', mol_file)
        molfile_to_params(mol_filepath, name)
        molfile_to_params_temps(mol_filepath, name)
    
    # 确保RESP文件夹存在
    if not os.path.exists(resp_folder):
        os.makedirs(resp_folder)
    
    # 生成resp文件
    for filename in os.listdir(gjf_folder):
        if filename.endswith('.log'):
            res = os.path.splitext(filename)[0]  # Get the base name without extension
            resp = os.path.join(resp_folder, res)  # RESP file path
            geom_log_path = os.path.join(gjf_folder, filename)

            # Run the generate_opt function
            generate_opt(geom_log_path, resp)
    
    # 处理原子类型调整
    n_value = names_list[0]
    covalent_context = None
    if covalent_target:
        context_key = f'{n_value}_1'
        covalent_context = covalent_contexts.get(context_key)
        if covalent_context is None:
            fail(f"Covalent atom mapping was not generated for {context_key}.")

    atom_type_adjust(resp_folder, covalent_context=covalent_context)
    
    # 计算capcharge
    calculate_capcharge(resp_folder)
    
    # 处理params文件
    process_params_file(n_value)
    
    mol2_filepath = os.path.join('RESP', f'{n_value}_1.mol2')
    params_filepath = os.path.join(f'{n_value}_temps.params')
 
    # Step 1: Calculate system charge from the actual CLI input file.
    system_charge = calculate_system_charge(input_file)
    print(f"System charge: {system_charge}")
 
    # Step 2: Read and process mol2 file
    included_atom_ids = (
        covalent_context['kept_params_atom_ids'] if covalent_context else None
    )
    atom_lines = read_mol2_file(mol2_filepath, included_atom_ids=included_atom_ids)
    charges_list = []
    for line in atom_lines:
        parts = line.split()
        charge = float(parts[-1])
        charges_list.append(charge)
    
    # Step 3: Adjust charges to ensure they sum to an integer
    if covalent_context:
        rounded_charges = [round(charge, 2) for charge in charges_list]
        print("Covalent mode: preserving rounded RESP charges without integer rebalancing.")
    else:
        rounded_charges = adjust_charges_to_integer(charges_list,system_charge)
    print(f"Charges list after adjustment: {rounded_charges}")
 
    # Step 4: Modify params file
    modify_params_file(params_filepath, rounded_charges)
    print(f"Modified {params_filepath} with rounded charges.")
 
    # Step 5: Update params file with temps params
    temps_filepath = os.path.join(f'{n_value}.params')
    update_params_file_with_temps(temps_filepath, params_filepath)
    print(f"Updated {params_filepath} with data from {temps_filepath}")

    if covalent_context:
        covalent_context = postprocess_covalent_params(
            temps_filepath,
            params_filepath,
            covalent_context,
        )
        context_path = f'{n_value}_covalent.json'
        with open(context_path, 'w') as context_file:
            json.dump(covalent_context, context_file, indent=2)
        print(f"Saved final covalent mapping to {context_path}")
    
    if covalent_context:
        print(
            "Covalent Rosetta params are complete. Skipping standalone GROMACS RTP "
            "generation: an arbitrary inter-residue crosslink requires a paired residue "
            "topology plus an explicit GROMACS crosslink/specbond definition."
        )
    else:
        process_mol2_with_args(n_value)

        # 生成top文件
        generate_top(resp_folder)

        resp = f"{res_rtp}_gromacs_prm/MOL.amb2gmx/MOL_GMX.top"

        # 生成rtp文件
        generate_rtp(resp)
    
    
    # 删除中间文件夹
    if clean == 1:
        shutil.rmtree('pdb_files', ignore_errors=True)
        shutil.rmtree('PDB_rearranged', ignore_errors=True)
        shutil.rmtree('GJF', ignore_errors=True)
        shutil.rmtree('mol', ignore_errors=True)
        shutil.rmtree('RESP', ignore_errors=True)
        for temporary_file in ('molecule.chk', 'fort.7'):
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
        print("Intermediate folders deleted.")
    else:
        organize_outputs(n_value, input_file)
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SMILES and convert to PDB files.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input SMILES file.')
    parser.add_argument('-n', '--names', required=True, nargs='+', help='Names for each SMILES (UAA if not specified).')
    parser.add_argument('-c', '--clean', type=int, choices=[0, 1], default=0, help='Clean intermediate folders: 1 to clean, 0 to keep (default).')
    parser.add_argument('--stage', choices=['all', 'gjf_only', 'from_logs'], default='all',
                        help='Pipeline stage: all runs local Gaussian, gjf_only stops after generating GJF files, from_logs resumes from completed GJF/*.log files.')
    parser.add_argument('--covalent', choices=COVALENT_TARGETS, default=None,
                        help='Generate a covalent Rosetta residue for a CYS/LYS/HIS/TYR adduct model.')
    parser.add_argument('--input-form', choices=['auto', 'free', 'capped'], default='auto',
                        help='Whether the SMILES is a free amino acid or an ACE-NCAA-NME capped model (default: auto).')
    parser.add_argument('--chirality', choices=['auto', 'L', 'D'], default='auto',
                        help='Polymer chirality override. Recommended for capped SMILES because @/@@ depends on atom order.')
    args = parser.parse_args()
    main(
        args.input,
        args.names,
        args.clean,
        args.stage,
        covalent_target=args.covalent,
        input_form=args.input_form,
        chirality=args.chirality,
    )
