"""Run with: python -m unittest discover -s tests -v."""
import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import ncaa
from pdb_io import Atom, parse_atoms
from gen_pos_file import positions, target_key
from gen_match_merge import parse_match_info, split_models, link_record, MatchInfo
from deduplicate import signature


def atom(serial,name,resname,chain,resnum,xyz):
    return Atom('ATOM',serial,name,resname,chain,resnum,'',*xyz,1.,0.,name[0])


class Scripts(unittest.TestCase):
    def test_covalent_link_fields(self):
        entry=ncaa.read_entry('LC4')
        atoms=[atom(1,'SG','CYF','A',279,(0,0,0)),atom(2,'CE2','LC4','B',5,(1.82,0,0))]
        line=link_record(atoms,MatchInfo('2CW','L1A','B',5,'CYW','A',279),entry)
        self.assertEqual((line[12:16].strip(),line[17:20],line[21],int(line[22:26])),('SG','CYF','A',279))
        self.assertEqual((line[42:46].strip(),line[47:50],line[51],int(line[52:56])),('CE2','LC4','B',5))
        self.assertEqual(float(line[73:78]),1.82)

    def test_four_digit_and_negative_pdb_fields(self):
        for num in [1000,-12]:
            a=atom(12345,'CA','ALA','A',num,(-123.456,45.678,-9.123))
            self.assertEqual(Atom.from_pdb_line(a.to_pdb_line()),a)

    def test_pos_uses_pose_order_not_pdb_numbers(self):
        atoms=[atom(1,'CA','CYS','A',1000,(0,0,0)),atom(2,'CA','ALA','B',15,(3,0,0)),atom(3,'CA','ALA','B',16,(20,0,0))]
        self.assertEqual(positions(atoms,('A',1000,''),'B',12),(1,[2]))
        self.assertEqual(target_key('A:1000B'),('A',1000,'B'))
        with self.assertRaises(ValueError): positions(atoms,('A',1000,''),'C',12)

    def test_duplicate_and_altloc_atoms_rejected(self):
        line=atom(1,'CA','ALA','B',1,(0,0,0)).to_pdb_line()
        with self.assertRaises(ValueError): parse_atoms([line,line])
        with self.assertRaises(ValueError): parse_atoms([line[:16]+'A'+line[17:]])

    def test_target_and_stub_roles(self):
        info=parse_match_info(['REMARK 666 MATCH TEMPLATE X 2CW 0 MATCH MOTIF A CYW 279 1 1',
                               'REMARK 666 MATCH TEMPLATE X 2CW 0 MATCH MOTIF B L1A 5 2 1'])
        self.assertEqual((info.target,info.stub,info.target_resseq),('CYW','L1A',279))

    def test_cloudpdb_preserves_global_remarks(self):
        text='REMARK 666 metadata\nMODEL        1\nATOM one\nENDMDL\nMODEL        2\nATOM two\nENDMDL\n'
        models=split_models(text)
        self.assertEqual(len(models),2)
        self.assertTrue(all(x[0]=='REMARK 666 metadata' for x in models))
        with self.assertRaises(ValueError):split_models('MODEL        1\nATOM incomplete\n')

    def test_all_twenty_atom_mappings_preserve_bonds(self):
        for name in sorted(ncaa.ANCHORS):
            with self.subTest(name=name):
                e=ncaa.read_entry(name)
                _,full,fb=ncaa.parameter_atoms(ncaa.PARAMETERS/'final'/f'{name}.params')
                covered=set()
                for group,residue,mapping in [('warheads',e.warhead,e.warhead_2_ncaa),('stubs',e.stub,e.stub_2_ncaa)]:
                    _,types,bonds=ncaa.parameter_atoms(ncaa.PARAMETERS/group/f'{residue}.params')
                    heavy={a for a,t in types.items() if not t.startswith('H') and t!='VIRT'}
                    for a in heavy:
                        aa=mapping.get(a,a);covered.add(aa)
                        self.assertIn(aa,full)
                        for b in bonds[a]&heavy:
                            self.assertIn(mapping.get(b,b),fb[aa],f'{name}: {a}-{b}')
                self.assertEqual(covered,{a for a,t in full.items() if not t.startswith('H') and t!='VIRT'})

    def test_dedup_ignores_headers_preserves_coordinates_and_chemistry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            line=atom(1,'CA','L1A','B',1,(0,0,0)).to_pdb_line()
            a=root/'first_LC4.pdb';b=root/'second_LC4.pdb';c=root/'second_LA5.pdb'
            a.write_text('REMARK first\n'+line+'END\n')
            b.write_text('REMARK second\n'+line+'END\n');c.write_text(b.read_text())
            self.assertEqual(signature(a),signature(b))
            self.assertNotEqual(signature(a),signature(c))
            b.write_text(atom(1,'CA','L1A','B',1,(.001,0,0)).to_pdb_line())
            self.assertNotEqual(signature(a),signature(b))

    def test_flags_include_current_assets_and_conformer(self):
        with tempfile.TemporaryDirectory(prefix='covmatch space ') as tmp:
            root=Path(tmp);pdb=root/'input model.pdb';pos=root/'input.pos'
            pdb.write_text(atom(1,'CA','CYS','A',1,(0,0,0)).to_pdb_line())
            pos.write_text('N_CST 2\n1: 1\n2: 2\n')
            r=subprocess.run([sys.executable,str(ROOT/'scripts/gen_match_flags.py'),'-s',str(pdb),'--pos',str(pos),'-n','ALL','-o',str(root/'jobs')],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            flags=list((root/'jobs').rglob('match.flags'));self.assertEqual(len(flags),20)
            for p in flags:self.assertIn('-enumerate_ligand_rotamers true',p.read_text())
            self.assertIn('PDB_ROTAMERS 3AW_conformers.pdb',(ncaa.PARAMETERS/'warheads/3AW.params').read_text())

    def test_dedup_keeps_job_ncaa_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'jobs_LM1_run';job.mkdir()
            pdb=job/'UM_candidate.pdb'
            pdb.write_text(atom(1,'CA','LFB','B',1,(0,0,0)).to_pdb_line())
            output=root/'unique'
            r=subprocess.run([sys.executable,str(ROOT/'scripts/deduplicate.py'),str(job),'-o',str(output)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            from gen_match_merge import detect_ncaa_from_path
            self.assertEqual(detect_ncaa_from_path(next(output.glob('*.pdb'))),'LM1')
            self.assertTrue(pdb.exists())

    def test_gaussian_final_segment_must_succeed(self):
        tree=ast.parse((ROOT/'scripts/R_G_parameterize.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='is_gaussian_job_completed')
        import re
        ns={'Path':Path,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'completion','exec'),ns)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.log'
            for text,expected in [('Normal termination',True),('Normal termination\n--Link1--\nError termination',False),('Normal termination\nEntering Link 1 =\n',False),('Normal termination\nStandard orientation:\n',False),('Error termination\nEntering Link 1 =\nNormal termination',True)]:
                p.write_text(text);self.assertEqual(ns['is_gaussian_job_completed'](p),expected,text)


if __name__=='__main__':
    unittest.main()
