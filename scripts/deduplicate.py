"""Copy unique Match or merged PDB structures without deleting inputs."""
import argparse
import csv
from pathlib import Path
import shutil
from gen_match_merge import collect_inputs, detect_ncaa_from_path, split_models
from pdb_io import parse_atoms


def signature(path):
    text=path.read_text()
    models=[]
    for lines in split_models(text):
        atoms=parse_atoms(lines)
        models.append(tuple(sorted((a.chain,a.resseq,a.icode,a.resname,a.name,a.element,a.x,a.y,a.z)
                                   for a in atoms)))
    if not any(models):
        raise ValueError(f'No atoms in {path}')
    links=tuple(sorted(l.strip() for l in text.splitlines() if l.startswith(('LINK  ','SSBOND'))))
    return detect_ncaa_from_path(path), tuple(models), links


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+')
    p.add_argument('-o','--outdir',required=True,type=Path)
    a=p.parse_args()
    files=collect_inputs(a.inputs)
    output=a.outdir.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Choose an empty output directory.')
    # Output must not fall within a recursively scanned input directory.
    for source in a.inputs:
        root=Path(source).resolve()
        if root.is_dir() and output.is_relative_to(root):
            p.error('Place the output outside input directories.')
    seen={}; rows=[]; selected=[]
    for path in files:
        key=signature(path)
        if key not in seen:
            candidate=detect_ncaa_from_path(path)
            prefix=(candidate+'_') if candidate else ''
            dest=output/f'{len(selected)+1:05d}_{prefix}{path.name}'
            seen[key]=(path,dest); selected.append((path,dest))
        kept,dest=seen[key]
        rows.append((str(path.resolve()),str(kept.resolve()),str(dest),'keep' if path==kept else 'duplicate'))
    output.mkdir(parents=True,exist_ok=True)
    for source,dest in selected:
        shutil.copyfile(source,dest)
    with (output/'deduplication.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['input','representative','output','status']); writer.writerows(rows)
    print(f'{len(files)} inputs; {len(selected)} unique structures; {len(files)-len(selected)} duplicates.')


if __name__=='__main__':
    main()
