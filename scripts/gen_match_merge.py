#!/usr/bin/env python3
"""Merge Rosetta Match outputs into complete covalent NCAA complexes."""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import ncaa


from pdb_io import Atom, parse_atoms

SUPPORTED_NCAA_RE = re.compile(r"(?<![A-Za-z0-9])(LA[5-8]|LC[4-7]|L[MOP][1-4])(?![A-Za-z0-9])")


@dataclass
class MatchInfo:
    warhead: str
    stub: str
    stub_chain: str
    stub_resseq: int
    target: str
    target_chain: str
    target_resseq: int

    @property
    def stub_key(self) -> tuple[str, int, str]:
        return (self.stub_chain, self.stub_resseq, "")

    @property
    def target_key(self) -> tuple[str, int, str]:
        return (self.target_chain, self.target_resseq, "")


def fail(message: str) -> None:
    banner = "\n" + "=" * 80 + f"\n[FATAL ERROR] {message}\n" + "=" * 80
    raise RuntimeError(banner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge CovMatch Rosetta Match PDB/CloudPDB outputs into full NCAA "
            "complexes and generate downstream relax constraints/scripts."
        )
    )
    parser.add_argument("inputs", nargs="*", help="PDB files or directories. Default: UM*.pdb in cwd.")
    parser.add_argument("-n", "--native_structure", metavar="PDB", default=None, help="Native PDB for RMSD metrics.")
    parser.add_argument("-s", "--nstruct_for_relax", type=int, metavar="N", default=10, help="Relax -nstruct, default 10.")
    parser.add_argument("--ncaa", default=None, help="Full NCAA name. Required when auto-detection is ambiguous.")
    parser.add_argument("-o", "--outdir", default="merge_outputs", help="Output directory.")
    parser.add_argument("--max-models", type=int, default=None, help="Limit models per input file for smoke tests.")
    parser.add_argument("--keep-h", action="store_true", help="Keep hydrogens. Default removes all hydrogens.")
    parser.set_defaults(rename_target=True)
    parser.add_argument("--rosetta-scripts-bin", default=os.environ.get("ROSETTA_SCRIPTS_BIN", "rosetta_scripts.default.linuxgccrelease"))
    parser.add_argument("--xml", default=None, help="RosettaScripts XML used with --relax.")
    parser.add_argument("--extra-improper-file", default=str(ncaa.PARAMETERS / "ncaa.tors"))
    parser.add_argument(
        "--params-dir",
        action="append",
        default=[],
        help="Directory searched for NCAA/target params. Can be repeated.",
    )
    parser.add_argument("--relax", action="store_true", help="Also generate a Relax command file; requires --xml.")
    parser.add_argument("--rotlib-dir", type=Path, help="Directory containing the external NCAA rotamer libraries.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing merged PDBs.")
    args = parser.parse_args()
    if args.relax and (not args.xml or not Path(args.xml).is_file()):
        parser.error("--relax requires an existing --xml protocol.")
    if args.nstruct_for_relax < 1 or (args.max_models is not None and args.max_models < 1):
        parser.error("Use positive structure/model counts.")
    return args


def split_models(text: str) -> list[list[str]]:
    lines = text.splitlines()
    if not any(line.startswith("MODEL ") for line in lines):
        return [lines]
    header = []
    models = []
    current = None
    for line in lines:
        if line.startswith("MODEL "):
            if current is not None:
                raise ValueError("MODEL without ENDMDL")
            current = list(header)
        elif line.startswith("ENDMDL"):
            if current is None:
                raise ValueError("ENDMDL without MODEL")
            models.append(current)
            current = None
        elif current is not None:
            current.append(line)
        elif not models:
            header.append(line)
    if current is not None:
        raise ValueError("Truncated final MODEL")
    return models


def parse_match_info(lines: Iterable[str]) -> MatchInfo:
    warhead = None
    stub = None
    target = None
    stub_chain = target_chain = None
    stub_resseq = target_resseq = None
    for line in lines:
        if not line.startswith("REMARK 666") or "MATCH TEMPLATE" not in line:
            continue
        parts = line.split()
        # REMARK 666 MATCH TEMPLATE X SYW 0 MATCH MOTIF B ABW 4 1 1
        if len(parts) < 12:
            continue
        template_res = parts[5]
        motif_chain = parts[9]
        motif_res = parts[10]
        motif_num = int(parts[11])
        if template_res in ncaa.warhead_lib:
            warhead = template_res
        if len(parts) < 14:
            continue
        role = int(parts[12])
        if role == 2 and motif_res in ncaa.stub_lib:
            stub = motif_res
            stub_chain = motif_chain
            stub_resseq = motif_num
        elif role == 1 and motif_res in ncaa.target_conj_atom:
            target = motif_res
            target_chain = motif_chain
            target_resseq = motif_num
    missing = [
        name for name, value in [
            ("warhead", warhead),
            ("stub", stub),
            ("stub_chain", stub_chain),
            ("stub_resseq", stub_resseq),
            ("target", target),
            ("target_chain", target_chain),
            ("target_resseq", target_resseq),
        ]
        if value is None
    ]
    if missing:
        fail(f"Could not parse Match REMARK 666 fields: missing {', '.join(missing)}")
    return MatchInfo(warhead, stub, stub_chain, stub_resseq, target, target_chain, target_resseq)


def collect_inputs(paths: list[str]) -> list[Path]:
    if not paths:
        return sorted(Path.cwd().glob("UM*.pdb"))
    files: list[Path] = []
    for item in paths:
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob("UM*.pdb")))
            files.extend(sorted(path.rglob("*.pdb")))
        elif path.is_file():
            files.append(path)
        else:
            fail(f"Input does not exist: {item}")
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def detect_ncaa_from_path(path: Path) -> str | None:
    matches = SUPPORTED_NCAA_RE.findall(path.name) or SUPPORTED_NCAA_RE.findall(path.parent.name)
    return matches[-1] if matches else None


def residue_order(records: list[Atom]) -> list[tuple[str, int, str, str]]:
    order = []
    seen = set()
    for atom in records:
        key = (*atom.residue_key(), atom.resname)
        if key not in seen:
            order.append(key)
            seen.add(key)
    return order


def rosetta_number(records: list[Atom], residue_key: tuple[str, int, str]) -> int:
    for idx, item in enumerate(residue_order(records), start=1):
        if item[:3] == residue_key:
            return idx
    fail(f"Could not find residue {residue_key} in merged records")


def load_param_atoms(name: str, search_dirs: list[Path]) -> set[str] | None:
    for directory in search_dirs:
        path = directory / f"{name}.params"
        if not path.exists():
            continue
        atoms = set()
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("ATOM"):
                parts = line.split()
                if len(parts) >= 2:
                    atoms.add(parts[1])
        return atoms
    return None


def default_param_dirs(extra: list[str]) -> list[Path]:
    dirs = [Path(item) for item in extra]
    for candidate in [ncaa.PARAMETERS / "final", ncaa.PARAMETERS / "targets"]:
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


def q(path: Path | str) -> str:
    return shlex.quote(str(path))


def merge_model(
    base_atoms: list[Atom],
    warhead_atoms: list[Atom],
    info: MatchInfo,
    mapping: ncaa.NCAA,
    keep_h: bool,
    rename_target: bool,
    param_atoms: set[str] | None,
) -> list[Atom]:
    if not warhead_atoms:
        fail("No warhead atoms found for model")
    stub_key = info.stub_key
    target_key = info.target_key
    merged: list[Atom] = []
    stub_insert_after = None
    seen_stub = False
    for atom in base_atoms:
        if not keep_h and atom.is_hydrogen():
            continue
        if atom.resname == info.warhead:
            continue
        new_atom = atom
        if atom.residue_key() == target_key and atom.resname == info.target and rename_target:
            new_atom = replace(atom, resname=ncaa.normalize_target_name(atom.resname), record="ATOM")
        if atom.residue_key() == stub_key and atom.resname == info.stub:
            new_atom = replace(new_atom, name=mapping.stub_2_ncaa.get(atom.name, atom.name),
                               resname=mapping.ncaa_name, record="ATOM")
            seen_stub = True
        merged.append(new_atom)
        if new_atom.residue_key() == stub_key and new_atom.resname == mapping.ncaa_name:
            stub_insert_after = len(merged)
    if not seen_stub or stub_insert_after is None:
        fail(f"Could not find stub residue {info.stub} {info.stub_chain}{info.stub_resseq}")

    mapped_warhead: list[Atom] = []
    for atom in warhead_atoms:
        if not keep_h and atom.is_hydrogen():
            continue
        if atom.name not in mapping.warhead_2_ncaa:
            fail(
                f"No warhead atom mapping for {mapping.ncaa_name}: "
                f"{info.warhead}.{atom.name}"
            )
        mapped_warhead.append(
            replace(
                atom,
                record="ATOM",
                name=mapping.warhead_2_ncaa[atom.name],
                resname=mapping.ncaa_name,
                chain=info.stub_chain,
                resseq=info.stub_resseq,
                icode="",
            )
        )

    merged[stub_insert_after:stub_insert_after] = mapped_warhead
    for idx, atom in enumerate(merged, start=1):
        atom.serial = idx

    if param_atoms is not None:
        ncaa_atoms = [atom.name for atom in merged if atom.residue_key() == stub_key and atom.resname == mapping.ncaa_name]
        extra = sorted(set(ncaa_atoms) - param_atoms)
        duplicates = sorted({name for name in ncaa_atoms if ncaa_atoms.count(name) > 1})
        if extra:
            fail(f"Merged {mapping.ncaa_name} atoms not present in params: {', '.join(extra)}")
        if duplicates:
            fail(f"Duplicate atoms in merged {mapping.ncaa_name}: {', '.join(duplicates)}")
    return merged


def link_record(records, info, mapping):
    target = next(a for a in records if a.residue_key() == info.target_key and a.name == 'SG')
    partner = next(a for a in records if a.residue_key() == info.stub_key and a.name == mapping.conj)
    line = list(' ' * 80)
    line[:6] = 'LINK  '
    for offset, atom in [(0, target), (30, partner)]:
        line[12+offset:16+offset] = f' {atom.name:<3}' if len(atom.name) < 4 else atom.name
        line[17+offset:20+offset] = f'{atom.resname:>3}'
        line[21+offset:22+offset] = atom.chain or ' '
        line[22+offset:26+offset] = f'{atom.resseq:4d}'
        line[26+offset:27+offset] = atom.icode or ' '
    distance = math.dist((target.x,target.y,target.z), (partner.x,partner.y,partner.z))
    line[73:78] = f'{distance:5.2f}'
    return ''.join(line).rstrip() + '\n'


def write_pdb(path: Path, records: list[Atom], source: Path, model_idx: int, mapping: ncaa.NCAA, info: MatchInfo) -> None:
    with path.open("w", newline="\n") as handle:
        handle.write(f"REMARK CovMatch merged from {source.name} model {model_idx}\n")
        handle.write(f"REMARK CovMatch NCAA {mapping.ncaa_name} warhead {mapping.warhead} stub {mapping.stub} target {mapping.target}\n")
        handle.write(link_record(records, info, mapping))
        for atom in records:
            handle.write(atom.to_pdb_line())
        handle.write("END\n")


def write_constraint(path: Path, records: list[Atom], info: MatchInfo, mapping: ncaa.NCAA) -> tuple[int, int, str, str]:
    target_key = info.target_key
    stub_key = info.stub_key
    target_num = rosetta_number(records, target_key)
    ncaa_num = rosetta_number(records, stub_key)
    target_conj, target_sub = ncaa.target_conj_atom[info.target]
    distance, angle_a, angle_b = mapping.cst_values
    with path.open("w", newline="\n") as handle:
        handle.write(f"AtomPair {target_conj} {target_num} {mapping.conj} {ncaa_num} HARMONIC {distance:.2f} 0.1\n")
        handle.write(
            f"Angle {target_conj} {target_num} {mapping.conj} {ncaa_num} {mapping.subconj} {ncaa_num} "
            f"HARMONIC {angle_a / 180.0 * math.pi:.4f} 0.1\n"
        )
        handle.write(
            f"Angle {target_sub} {target_num} {target_conj} {target_num} {mapping.conj} {ncaa_num} "
            f"HARMONIC {angle_b / 180.0 * math.pi:.4f} 0.1\n"
        )
        binder_chain = info.stub_chain
        for atom in records:
            if atom.name != "CA":
                continue
            ros_num = rosetta_number(records, atom.residue_key())
            if atom.chain != binder_chain or abs(ros_num - ncaa_num) > 1:
                pdb_res = f"{atom.resseq}{atom.chain}"
                handle.write(
                    f"CoordinateConstraint CA {pdb_res} CA {pdb_res} "
                    f"{atom.x:.3f} {atom.y:.3f} {atom.z:.3f} HARMONIC 0 1\n"
                )
    return target_num, ncaa_num, target_conj, mapping.conj


def find_params_for_script(names: Iterable[str], search_dirs: list[Path]) -> list[Path]:
    found = []
    missing = []
    for name in names:
        path = None
        for directory in search_dirs:
            candidate = directory / f"{name}.params"
            if candidate.exists():
                path = candidate
                break
        if path is None:
            missing.append(name)
        else:
            found.append(path)
    if missing:
        fail(f"Missing complete params: {', '.join(missing)}")
    return found


def append_relax_command(
    handle,
    pdb_path: Path,
    cst_path: Path,
    args: argparse.Namespace,
    info: MatchInfo,
    mapping: ncaa.NCAA,
    target_num: int,
    ncaa_num: int,
    target_conj: str,
    conj_atom: str,
    param_paths: list[Path],
) -> None:
    if not args.relax:
        return
    xml = str(Path(args.xml).resolve())
    parts = [
        q(args.rosetta_scripts_bin),
        "-s", q(pdb_path),
        "-parser:protocol", q(xml),
        "-nstruct", str(args.nstruct_for_relax),
        "-crystal_refine",
        "-parser:script_vars",
        f"ncaa_num={ncaa_num}",
        f"upstream_num={target_num}",
        f"target_chain={info.target_chain}",
        f"binder_chain={info.stub_chain}",
        q(f"cst_file_name={cst_path}"),
        f"upstream_conj={target_conj}",
        f"conj_atom={conj_atom}",
        "-mute", "all",
    ]
    if args.rotlib_dir:
        parts.extend(["-extra_rotlib_path", q(args.rotlib_dir.resolve())])
    parts.extend(["-out:path:all", q(pdb_path.parent.parent / "relaxed")])
    if Path(args.extra_improper_file).exists():
        parts.extend(["-extra_improper_file", q(Path(args.extra_improper_file).resolve())])
    if param_paths:
        parts.extend(["-extra_res_fa", *[q(path) for path in param_paths]])
    if args.native_structure:
        parts.extend(["-in:file:native", q(Path(args.native_structure).resolve())])
    handle.write(" ".join(parts) + "\n")


def process_file(path: Path, args: argparse.Namespace, outdir: Path, param_dirs: list[Path], relax_handle, summary_handle) -> int:
    text = path.read_text(errors="replace")
    models = split_models(text)
    if not models:
        fail(f"No PDB models/atoms found in {path}")
    info = parse_match_info(models[0])
    ncaa_name = args.ncaa or detect_ncaa_from_path(path)
    mapping = ncaa.find_mapping(info.warhead, info.stub, info.target, ncaa_name)
    if mapping is None:
        fail(
            f"No NCAA mapping for file={path}, ncaa={ncaa_name}, "
            f"warhead={info.warhead}, stub={info.stub}, target={info.target}"
        )
    base_first = [atom for atom in parse_atoms(models[0]) if atom.resname != info.warhead]
    param_atoms = load_param_atoms(mapping.ncaa_name, param_dirs)
    if param_atoms is None:
        fail(f"Missing complete params for {mapping.ncaa_name}")
    rows, types, bonds = ncaa.parameter_atoms(next(d / f"{mapping.ncaa_name}.params" for d in param_dirs if (d / f"{mapping.ncaa_name}.params").is_file()))
    heavy_atoms = {name for name, kind in types.items() if not kind.startswith("H") and kind != "VIRT"}
    param_names = [mapping.ncaa_name, ncaa.normalize_target_name(info.target) if args.rename_target else info.target]
    param_paths = find_params_for_script(param_names, param_dirs)

    produced = 0
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    for model_idx, model_lines in enumerate(models, start=1):
        if args.max_models is not None and produced >= args.max_models:
            break
        model_atoms = parse_atoms(model_lines)
        warhead_atoms = [atom for atom in model_atoms if atom.resname == info.warhead]
        if not warhead_atoms:
            continue
        has_scaffold = any(atom.residue_key() == info.stub_key for atom in model_atoms)
        base_atoms = [atom for atom in model_atoms if atom.resname != info.warhead] if has_scaffold else base_first
        merged = merge_model(base_atoms, warhead_atoms, info, mapping, args.keep_h, args.rename_target, param_atoms)
        actual = {a.name for a in merged if a.residue_key() == info.stub_key}
        if heavy_atoms - actual:
            fail(f"Missing heavy atoms in {mapping.ncaa_name}: {sorted(heavy_atoms - actual)}")
        output_pdb = outdir / "pdb" / f"Merge_{safe_stem}_model{model_idx:03d}_{mapping.ncaa_name}.pdb"
        output_cst = outdir / "cst" / f"{output_pdb.stem}.cst"
        if not args.dry_run:
            write_pdb(output_pdb, merged, path, model_idx, mapping, info)
            target_num, ncaa_num, target_conj, conj_atom = write_constraint(output_cst, merged, info, mapping)
            append_relax_command(
                relax_handle,
                output_pdb,
                output_cst,
                args,
                info,
                mapping,
                target_num,
                ncaa_num,
                target_conj,
                conj_atom,
                param_paths,
            )
        else:
            target_num = rosetta_number(merged, info.target_key)
            ncaa_num = rosetta_number(merged, info.stub_key)
        summary_handle.write(
            "\t".join([
                str(path),
                str(model_idx),
                mapping.ncaa_name,
                info.warhead,
                info.stub,
                info.target,
                f"{info.stub_chain}{info.stub_resseq}",
                f"{info.target_chain}{info.target_resseq}",
                str(output_pdb if not args.dry_run else "DRY_RUN"),
                str(target_num),
                str(ncaa_num),
            ]) + "\n"
        )
        produced += 1
    return produced


def main() -> None:
    args = parse_args()
    inputs = [p.resolve() for p in collect_inputs(args.inputs)]
    if not inputs:
        fail("No input PDB files found.")
    outdir = Path(args.outdir).resolve()
    if outdir.exists() and any(outdir.iterdir()) and not args.dry_run:
        fail("Choose an empty output directory.")
    param_dirs = default_param_dirs(args.params_dir)
    if not args.dry_run:
        (outdir / "pdb").mkdir(parents=True, exist_ok=True)
        (outdir / "cst").mkdir(parents=True, exist_ok=True)
        (outdir / "logs").mkdir(parents=True, exist_ok=True)
        if args.relax:
            (outdir / "relaxed").mkdir(exist_ok=True)
    if len({p.stem for p in inputs}) != len(inputs):
        fail("Duplicate input filenames; deduplicate or rename inputs first.")
    summary_path = outdir / "merge_summary.tsv"
    relax_path = outdir / "run_relax.sh"
    if args.dry_run:
        summary_handle = open(os.devnull, "w")
        relax_handle = open(os.devnull, "w")
    else:
        summary_handle = summary_path.open("w", newline="\n")
        relax_handle = relax_path.open("w", newline="\n") if args.relax else open(os.devnull, "w")
        relax_handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        summary_handle.write(
            "input_pdb\tmodel\tncaa\twarhead\tstub\ttarget\tstub_site\ttarget_site\tmerged_pdb\ttarget_rosetta_num\tncaa_rosetta_num\n"
        )
    total = 0
    try:
        for path in inputs:
            count = process_file(path, args, outdir, param_dirs, relax_handle, summary_handle)
            print(f"[OK] {path}: merged {count} model(s)")
            total += count
    finally:
        summary_handle.close()
        relax_handle.close()
    if not args.dry_run:
        if args.relax:
            relax_path.chmod(0o755)
        print(f"[DONE] merged_models={total}")
        print(f"[DONE] pdb_dir={outdir / 'pdb'}")
        print(f"[DONE] cst_dir={outdir / 'cst'}")
        print(f"[DONE] summary={summary_path}")
        if args.relax:
            print(f"[DONE] relax_script={relax_path}")
    else:
        print(f"[DRY-RUN DONE] mergeable_models={total}")


if __name__ == "__main__":
    main()
