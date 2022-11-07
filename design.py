"""Design a protein sequence given the backbone."""
import argparse
import os
from random import sample
import torch
from Bio.PDB import PDBParser, FastMMCIFParser, PDBIO
from pe.common import residue_constants as rc
from pe.common import convert
from pe.model import modules
import evaluator


def identity(str1, str2):
  c = 0
  for a, b in zip(str1, str2):
    if a == b:
      c += 1
  return c / len(str1)

def read_constrain_file(constrain):
  result = {}
  with open(constrain) as fin:
    for line in map(str.rstrip, fin):
      tokens = line.split()
      index = int(tokens[0]) - 1
      resi = tokens[1]
      result[index] = resi
  return result

def design(body, file_dir, constrain=None, log_file='', interval=1, args={}):
  if log_file:
    log_fin = open(log_file, 'w', encoding='utf-8')

  file_dir_root, file_type = os.path.splitext(file_dir)
  pid = os.path.basename(file_dir_root)
  if file_type not in ('.pdb', '.cif'):
    raise TypeError(f'Only .pdb and .cif type are supported, get {file_type}')

  protein_parser = PDBParser(QUIET=True) if file_type == '.pdb'\
    else FastMMCIFParser(QUIET=True)
  structure = protein_parser.get_structure('none', file_dir)
  models = list(structure.get_models())
  if len(models) != 1:
    raise ValueError(
      f'Only single model PDBs are supported. Found {len(models)} models.')
  model = models[0]

  main_chain_dir = os.path.join(args.cache_dir, f'{pid}_m.pdb')
  convert.reorder_model(model, 1)

  original_residues = []
  init_residues = []
  chains = list(model.get_chains())
  for chain in chains:
    for res in chain.get_residues():
      original_residues.append(rc.restype_3to1.get(res.resname, 'X'))
      res.resname = sample(list(rc.restype_1to3.values()), k=1)[0]
      init_residues.append(rc.restype_3to1.get(res.resname, 'X'))
  original_residues = ''.join(original_residues)
  io = PDBIO()
  io.set_structure(structure)
  io.save(main_chain_dir, convert.CustomSelect(only_main=True))

  if log_file:
    init_residues = ''.join(init_residues)
    log_fin.write(init_residues)
    log_fin.write('\n')

  seqs = []
  accs = []
  if args.strict:
    len_stage_2 = len(original_residues) * 3
    ks = [1] * len_stage_2
  else:
    len_stage_1 = int(len(original_residues)/ 1)
    len_stage_2 = int(len(original_residues) / 5)
    ks = [5] * len_stage_1 + [1] * len_stage_2

  for i, k in enumerate(ks):
    eval_result = evaluator.evaluate(body, main_chain_dir, return_logits=True,
      radius=3.5, return_is_optimal=True)
    acc = eval_result['accuracy']

    if i > 0:
      accs.append(acc)

    logits = eval_result['logits']
    new_index = logits.argmax(dim=-1)
    if constrain is not None:
      for i, r in constrain.items():
        new_index[i] = rc.resname_to_idx[rc.restype_1to3[r]]
    new_res = ''.join(rc.restypes_with_x[index] for index in new_index.tolist())
    structure = protein_parser.get_structure('none', main_chain_dir)
    model = list(structure.get_models())[0]
    chains = list(model.get_chains())

    non_optimal_list =\
      (~ eval_result['is_optimal']).nonzero().squeeze(dim=1).tolist()
    index_to_change = sample(non_optimal_list, min(len(non_optimal_list), k))
    if constrain is not None:
      index_to_change.extend(list(constrain.keys()))

    for j in index_to_change:
      old_residues = list(chains[0].get_residues())
      old_residues[j].resname = rc.restype_1to3.get(new_res[j], 'UNK')

    new_res = ''.join(
      rc.restype_3to1.get(res.resname, 'X') for res in old_residues)
    identity_ = identity(new_res, original_residues)
    seqs.append(new_res)

    io.set_structure(structure)
    io.save(main_chain_dir)

  seqs.pop()

  accs = torch.tensor(accs)
  index_chosen = accs.argmax().item()

  result = {}
  result['seq'] = seqs[index_chosen]

  if log_file:
    log_fin.close()
  return result


def main():
  num_class = len(rc.restypes_with_x)
  body = modules.Transformer(46, num_class, 256, nhead=16, nlayer=3,
    device=args.device)
  stored = torch.load(args.store_dir, map_location='cpu')
  body.load_state_dict(stored['model_state_dict'])

  input_files = []
  if os.path.isdir(args.input_path):
    for file in os.listdir(args.input_path):
      input_files.append(os.path.join(args.input_path, file))
  else:
    input_files.append(args.input_path)
  input_files = sorted(input_files)

  files_to_process = []
  for file in input_files:
    if os.path.splitext(file)[-1] in ('.cif', '.pdb'):
      files_to_process.append(file)
  print('Files to process:', len(files_to_process))

  constrain=None
  if args.constrain:
    constrain=read_constrain_file(args.constrain)

  for file in files_to_process:
    pid = os.path.splitext(os.path.basename(file))[0]
    for i in range(args.index, args.index + args.num_seq):
      print(f'Designing {pid}_{i}')
      log_file_path = ''
      if args.log:
        log_file_path = os.path.join(args.cache_dir, f'{pid}_{i}.log')
      result = design(body, file, constrain, log_file=log_file_path,
        interval=args.interval, args=args)
      with open(os.path.join(
          args.output_path, f'{pid}_{i}.fasta'), 'w', encoding='utf-8') as fout:
        fout.write(f'>{pid}\n')
        fout.write(f'{result["seq"]}\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('store_dir', type=str)
  parser.add_argument('input_path', type=str, help='A dir or cif/pdb file')
  parser.add_argument('-c', '--cache_dir', type=str, default='.')
  parser.add_argument('-o', '--output_path', type=str, default='.')
  parser.add_argument('-d', '--device', type=str, default='cpu')
  parser.add_argument('-C', '--constrain', type=str, help='1-indexed')
  parser.add_argument('-S', '--strict', action='store_true')
  parser.add_argument('-L', '--log', action='store_true')
  parser.add_argument('-I', '--interval', type=int, default=1,
    help='The interval to print a intermediate result to log.')
  parser.add_argument('-n', '--num_seq', type=int, default=1,
    help='number of fasta per structure')
  parser.add_argument('-i', '--index', type=int, default=1, help='start index')
  args = parser.parse_args()
  main()
