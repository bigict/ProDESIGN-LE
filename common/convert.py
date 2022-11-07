"""Some operation to a pdb/cif file."""
import argparse
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio import SeqIO

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

class CustomSelect(Select):
  """Selecter of pdb file"""
  def __init__(self, chain_id=None, only_main=False, skip_het=False) -> None:
    super().__init__()
    self.only_main = only_main
    self.chain_id = chain_id
    self.skip_het = skip_het

  def accept_chain(self, chain):
    if self.chain_id is not None:
      return chain.id in self.chain_id
    else:
      return True

  def accept_atom(self, atom):
    if self.only_main:
      return atom.name in ['N', 'C', 'CA']
    else:
      return True

  def accept_residue(self, residue):
    if self.skip_het:
      return residue.id[0] == ' '
    else:
      return True

def reorder_model(model, start_index):
  big_number = 100000
  for chain in model.get_chains():
    for residue in chain.get_residues():
      old_id = residue.id
      new_id = (old_id[0], old_id[1] + big_number, old_id[2])
      residue.id = new_id

    index = start_index
    for residue in chain.get_residues():
      old_id = residue.id
      new_id = (old_id[0], index, old_id[2])
      residue.id = new_id
      index += 1
  return model

def main():
  input_type = os.path.splitext(args.input_file)[-1]
  assert input_type in ('.cif', '.pdb')
  file_parser = PDBParser(QUIET=True) if input_type == '.pdb' else\
    MMCIFParser(QUIET=True)
  protein_structure = file_parser.get_structure('none', args.input_file)
  if args.list_chain:
    model = list(protein_structure.get_models())[0]
    chains = model.get_chains()
    for chain in chains:
      print(chain.id, len(list(chain.get_residues())))
  elif os.path.splitext(args.output_file)[-1] == '.pdb':
    model = list(protein_structure.get_models())[0]
    if args.start_index is not None:
      model = reorder_model(model, args.start_index)
    io = PDBIO()
    io.set_structure(protein_structure)
    io.save(args.output_file, select=CustomSelect(
      args.chain, args.only_main, args.skip_het
    ))
  else:
    seqs = []
    pid = os.path.splitext(os.path.basename(args.input_file))[0]
    model = list(protein_structure.get_models())[0]
    chains = model.get_chains()
    for chain in chains:
      seq_str = ''.join((restype_3to1.get(ele.get_resname(), 'X')
        for ele in chain.get_residues()))
      seq = SeqRecord(Seq(seq_str), id=f'{pid}|{chain.id}')
      seqs.append(seq)
    SeqIO.write(seqs, args.output_file, 'fasta')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file')
  parser.add_argument('output_file')
  parser.add_argument('-c', '--chain', nargs='*')
  parser.add_argument('-M', '--only_main', action='store_true')
  parser.add_argument('-l', '--list_chain', action='store_true')
  parser.add_argument('-s', '--skip_het', action='store_true')
  parser.add_argument('-S', '--start_index', type=int)
  args = parser.parse_args()
  main()
