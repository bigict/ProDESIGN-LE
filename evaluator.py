import argparse
from statistics import mean
import torch
from torch import nn
from Bio.PDB import PDBIO, StructureBuilder
from pe.data import loader, make_database
from pe.model import preprocess
from pe.common import util
from pe.common import residue_constants as rc
from pe.model import modules


def get_confidence(logits):
  """(B,) <- (B, C)"""
  probs = torch.softmax(logits, dim=-1).detach()  # (B, C)
  confidencies = (probs * torch.log(probs)).sum(-1)  # (B, )
  return confidencies


def get_acc(output, target, topk=1, reduction='mean'):
  assert reduction in ('mean', 'sum')
  batch_size = len(target)
  max_index = torch.topk(output, topk)[1]  # (B, k) <- (B, C)
  bingo = (max_index == target.unsqueeze(-1))  # (B, k)
  num_correct = bingo.any(dim=1).sum()  # () <- (B, k)
  if reduction=='sum':
    return num_correct
  else:
    return num_correct / batch_size

def evaluate(model, validate_file, topk=1, return_logits=False, by_chain=False,
    radius=3, cache_suffix='', cache_name=None, return_is_optimal=False, top_k=5,
    mask_type=False):

  result = {}
  model.eval()
  if return_logits:
    logits = []

  validate = loader.get_loader(
    make_database.make_dataset(validate_file, radius=radius,
      cache_suffix=cache_suffix,cache_name=cache_name),
    device=util.get_model_device(model), batch_size=1000)
  total_loss = 0
  losses = []
  is_optimal = []
  total_loss_by_chain = {}
  total_acc = 0
  n_sample = 0

  process = preprocess.PreProcess()
  loss = nn.CrossEntropyLoss(reduction='none')

  for batch in validate:
    processed = process(batch['feature'])
    if mask_type:
      processed[..., :21] = 0
    output = model(processed, mask=batch['mask'])
    acc = get_acc(output, batch['label']['target_name'], topk=topk,
      reduction='sum')
    if return_is_optimal:
      is_optimal.append(output.argmax(dim=1) == batch['label']['target_name'])
    l = loss(output, batch['label']['target_name'])

    total_loss += l.sum().detach().item()
    total_acc += acc.detach().item()
    if by_chain:
      for i, chain in enumerate(batch['feature']['target_chain_list']):
        if chain in total_loss_by_chain:
          total_loss_by_chain[chain].append(l[i].item())
        else:
          total_loss_by_chain[chain] = [l[i].item()]

    n_sample += processed.size(0)
    if return_logits:
      logits.append(output)

  if by_chain:
    mean_loss_by_chain = {}
    for chain, losses in total_loss_by_chain.items():
      mean_loss_by_chain[chain] = mean(losses)
      total_loss_by_chain[chain] = sum(losses)

  result['loss'] = total_loss/n_sample
  if by_chain:
    result['total_loss_by_chain'] = total_loss_by_chain
    result['mean_loss_by_chain'] = mean_loss_by_chain
  result['accuracy'] = total_acc/n_sample
  if return_is_optimal:
    result['is_optimal'] = torch.cat(is_optimal)
  if return_logits:
    result['logits'] = torch.cat(logits)
  return result
