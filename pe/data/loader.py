"""Dataloader"""
import functools
from torch.utils.data import DataLoader
import torch
from pe.common import residue_constants as rc
from pe.data.make_database import NeighborDataset


def collate_fn(data, device='cpu'):
  lengths = [len(sample['feature']['neighbor_name']) for sample in data]
  batch_size = len(lengths)
  batch_length = max(lengths)

  kw = {'device': device}

  mask = torch.zeros((batch_size, batch_length), dtype=torch.bool, **kw)
  neighbor_name = torch.zeros_like(mask, dtype=torch.long)
  neighbor_chain = torch.zeros_like(neighbor_name)
  neighbor_id = torch.zeros_like(mask, dtype=torch.long)
  neighbor_b_factor = torch.zeros((batch_size, batch_length, 3),
    dtype=torch.float, **kw)
  neighbor_main_chain_coord = torch.zeros(
    (batch_size, batch_length, 3, 3), dtype=torch.float, **kw
  )
  target_id = torch.zeros(batch_size, dtype=torch.long, **kw)
  target_chain = torch.zeros(batch_size, dtype=torch.long, **kw)
  target_chain_list = [''] * batch_size
  target_name = torch.zeros(batch_size, dtype=torch.long, **kw)
  target_main_chain_coord = torch.zeros((batch_size, 3, 3), **kw)
  target_b_factor = torch.zeros((batch_size, 3), **kw)

  for i, sample in enumerate(data):
    feature, label = sample['feature'], sample['label']
    mask[i, :lengths[i]] = True
    neighbor_name[i, :lengths[i]] = torch.tensor(list(map(
      lambda x: rc.resname_to_idx.get(x, len(rc.restypes)),
      feature['neighbor_name'])))
    neighbor_chain[i, :lengths[i]] =\
      torch.tensor(list(map(hash, feature['neighbor_chain'])))
    neighbor_id[i, :lengths[i]] = torch.tensor(feature['neighbor_id'])
    neighbor_b_factor[i, :lengths[i], ...] = feature['neighbor_b_factor']
    neighbor_main_chain_coord[i, :lengths[i], ...] =\
      feature['neighbor_main_chain_coord']
    target_id[i] = feature['target_id']
    target_chain[i] = hash(feature['target_chain'])
    target_chain_list[i] = feature['target_chain']
    target_main_chain_coord[i] = feature['target_main_chain_coord']
    target_name[i] = rc.resname_to_idx.get(label['target_name'],
      len(rc.restypes))
    target_b_factor[i] = label['target_b_factor']

  return {
    'feature':
      {'neighbor_name': neighbor_name,
        'neighbor_id': neighbor_id,
        'neighbor_chain': neighbor_chain,
        'neighbor_main_chain_coord': neighbor_main_chain_coord,
        'neighbor_b_factor': neighbor_b_factor,
        'target_main_chain_coord': target_main_chain_coord,
        'target_id': target_id,
        'target_chain': target_chain,
        'target_chain_list': target_chain_list
      },
    'label':
      {'target_name': target_name,
        'target_b_factor': target_b_factor
      },
    'mask': mask
    }

def get_loader(dataset: NeighborDataset, device='cpu', **kwargs):
  collate_fn_ = functools.partial(collate_fn, device=device)
  kwargs.update(dict(collate_fn=collate_fn_))
  return DataLoader(dataset, **kwargs)
