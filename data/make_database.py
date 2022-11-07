"""Make database from pdb files"""

import os
import pickle

import torch
from torch.utils.data import Dataset, ConcatDataset
from einops import rearrange
from Bio.PDB import PDBParser, FastMMCIFParser


class NeighborRecord:
  def __init__(self, file_path: str) -> None:

    file_type = os.path.splitext(file_path)[-1]
    if file_type not in ('.pdb', '.cif'):
      raise TypeError(f'Only .pdb and .cif type are supported, get {file_type}')

    parser = PDBParser(QUIET=True) if file_type == '.pdb'\
      else FastMMCIFParser(QUIET=True)
    structure = parser.get_structure('none', file_path)
    models = list(structure.get_models())
    if len(models) != 1:
      raise ValueError(
          f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    residue_infos = []
    b_factors = []
    main_chain_coords = []
    gloabl_res_id = []

    for chain in model.get_chains():
      for res in chain:
        chain_id_hash = hash(chain.id)

        if res.id[2] != ' ':
          raise ValueError(
            f'PDB contains an insertion code at chain {chain.id} and residue '
            f'index {res.id[1]}. These are not supported.')

        if res.id[0] != ' ':
          # This is not ordinary reisdue of the chain.
          continue

        main_chain_atom_names = ['N', 'CA', 'C']
        res_info = (chain.id, res.id[1], res.resname)
        b_factor = torch.zeros(3)
        main_chain_coord = torch.zeros(3, 3)
        atom_fill_flag = torch.zeros(3, dtype=torch.bool)
        for i, atom in enumerate(res.get_atoms()):
          if atom.name not in main_chain_atom_names:
            continue
          index = main_chain_atom_names.index(atom.name)
          main_chain_coord[index] = torch.tensor(atom.get_coord())
          b_factor[index] = atom.get_bfactor()
          atom_fill_flag[index] = True
          if atom_fill_flag.all():
            break
        if not atom_fill_flag.all():
          print(chain.id, res.id[1],
            list(zip(main_chain_atom_names, atom_fill_flag)))
          raise ValueError('Imcomplete main chain atom set')

        residue_infos.append(res_info)
        b_factors.append(b_factor)
        main_chain_coords.append(main_chain_coord)
        gloabl_res_id.append(chain_id_hash + res.id[1])

    gloabl_res_id = torch.tensor(gloabl_res_id)
    b_factors = torch.stack(b_factors)
    main_chain_coords = torch.stack(main_chain_coords)
    ca_coords = main_chain_coords[:, 1, :]
    ca_coords = rearrange(ca_coords, 'L d -> () L d', d=3)
    ca_mutual_distance = torch.cdist(ca_coords, ca_coords)[0]
    res_id_distance = gloabl_res_id.unsqueeze(-1) - gloabl_res_id.unsqueeze(0)

    self.residue_infos = residue_infos
    self.b_factors = b_factors
    self.main_chain_coords = main_chain_coords
    self.ca_mutual_distance = ca_mutual_distance
    self.res_id_distance = res_id_distance
    self.structure = structure

  def get_residue_neighbors(self, radius=3, max_index_diff=3):
    is_neighbor = self.ca_mutual_distance < (radius ** 2)
    is_neighbor_2 = torch.abs(self.res_id_distance) <= max_index_diff
    is_neighbor = is_neighbor | is_neighbor_2
    result = {}
    for i in range(len(self.residue_infos)):
      neighbor_indexs = is_neighbor[i].nonzero(as_tuple=True)[0]
      neighbor_distances = self.ca_mutual_distance[i][neighbor_indexs]
      neighbor_indexs = neighbor_indexs[neighbor_distances.argsort()]
      result[i] = neighbor_indexs
    return result

  def store_cache_to(self, file_name='', radius=3, to_return=False):
    cache = {'residue_info': self.residue_infos,
         'neighbor_index': self.get_residue_neighbors(radius),
         'b_factor': self.b_factors,
         'main_chain_coord': self.main_chain_coords,
         'radius': radius}
    if file_name:
      with open(file_name, 'wb') as fout:
        pickle.dump(cache, fout)
    if to_return:
      return cache


class NeighborDataset(Dataset):

  def __init__(self, file_path, radius=3, cache_suffix=''):
    super().__init__()

    cache_path =\
      f'{os.path.splitext(file_path)[0]}_{cache_suffix}.pkl' if cache_suffix\
        else ''
    if not os.path.exists(cache_path):
      db = NeighborRecord(file_path)
      cache = db.store_cache_to(cache_path, to_return=True, radius=radius)
    else:
      with open(cache_path, 'rb') as fin:
        cache = pickle.load(fin)
    self.radius = radius
    self.data = cache
    self.valid_sample_index = [i for i in cache['neighbor_index'].keys()
      if len(cache['neighbor_index'][i]) > 1]

  def __len__(self):
    return len(self.valid_sample_index)

  def __getitem__(self, index):
    index = self.valid_sample_index[index]
    label_index = self.data['neighbor_index'][index][0]
    target_chain, target_id, target_name =\
      self.data['residue_info'][label_index]
    neighbor_indexs = self.data['neighbor_index'][index][1:]
    main_chain_coord = self.data['main_chain_coord']
    chains, ids, residue_names =\
      zip(*[self.data['residue_info'][index] for index in neighbor_indexs])
    return {
      'feature':
        {'neighbor_name': residue_names,
         'neighbor_id': ids,
         'neighbor_chain': chains,
         'neighbor_main_chain_coord': main_chain_coord[neighbor_indexs, ...],
         'neighbor_b_factor': self.data['b_factor'][neighbor_indexs, ...],
         'target_main_chain_coord': main_chain_coord[label_index],
         'target_id': target_id,
         'target_chain': target_chain
        },
      'label':
        {'target_name': target_name,
         'target_b_factor': self.data['b_factor'][label_index]
        }
    }


def make_dataset(path, store_ds_cache=True, cache_name=None,
  pid_black_list=set(), radius=3, cache_suffix=''):
  if isinstance(path, list):
    return ConcatDataset([make_dataset(
      ele, store_ds_cache=store_ds_cache, cache_name=cache_name, radius=radius,
      pid_black_list=pid_black_list, cache_suffix=cache_suffix)
      for ele in path])
  if os.path.isdir(path):
    if cache_name:
      ds_pkl_path = os.path.join(path, f'{cache_name}.ds_pkl')
      if os.path.exists(ds_pkl_path):
        with open(ds_pkl_path, 'rb') as fin:
          return pickle.load(fin)
    datasets = []
    counter = 1
    print('Constructing datasests...')
    for file in filter(
        lambda x: os.path.splitext(x)[-1] in ('.pdb', '.cif'),
        os.listdir(path)):
      try:
        print(counter, file)
        pid = os.path.splitext(file)[0]
        if pid in pid_black_list:
          print(f'Skip black list item {pid}.')
        else:
          datasets.append(
            NeighborDataset(os.path.join(path, file), radius=radius,
              cache_suffix=cache_suffix))
      except ValueError:
        pass
      counter += 1
    db = ConcatDataset(datasets)
    if cache_name:
      with open(ds_pkl_path, 'wb') as fout:
        pickle.dump(db, fout)
    return db
  elif os.path.splitext(path)[-1] == '.ds_pkl':
    with open(path, 'rb') as fin:
      return pickle.load(fin)
  else:
    return NeighborDataset(path, radius=radius, cache_suffix=cache_suffix)
