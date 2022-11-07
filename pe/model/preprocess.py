"""Prepare actual input for N.N."""
import torch
from torch import nn
import torch.nn.functional as func
from einops import rearrange

from pe.common import residue_constants as rc
from pe.common import r3


def relative_one_hot(offset, max_relative_feature):
  clamped = torch.clamp(offset + max_relative_feature,
      min=0, max=2 * max_relative_feature)
  return func.one_hot(
    clamped,
    num_classes=2 * max_relative_feature + 1)


class PreProcess(nn.Module):

  def __init__(self,
      max_relative_feature=5, seq_feat_dim=128, pair_feat_dim=128):
    super().__init__()
    self.max_relative_feature = max_relative_feature
    self.seq_feat_dim = seq_feat_dim
    self.pair_feat_dim = pair_feat_dim

  def forward(self, feat):
    # Make seq feat
    aa_type = func.one_hot(feat['neighbor_name'],
      num_classes=len(rc.restypes) + 1).float()

    same_chain_with_target = (feat['neighbor_chain'] ==\
      rearrange(feat['target_chain'], 'B -> B ()')).type(torch.float)

    neigbor_position_id = feat['neighbor_chain'] + feat['neighbor_id']
    target_position_id = feat['target_chain'] + feat['target_id']
    offset = neigbor_position_id - target_position_id.unsqueeze(-1)
    rel_pos_to_target = relative_one_hot(offset, self.max_relative_feature)

    is_senpai = \
      (feat['neighbor_chain'] == feat['target_chain'].unsqueeze(-1)) &\
      (feat['neighbor_id'] < feat['target_id'].unsqueeze(-1))
    is_senpai = is_senpai.type(torch.float)

    neighbor_main_chain_coord = feat['neighbor_main_chain_coord']
    neighbor_frame = r3.vec2transform(neighbor_main_chain_coord)
    target_frame = r3.vec2transform(feat['target_main_chain_coord'])
    neighbor_frame_relative = torch.einsum('Bij,BLjk->BLik',
      r3.transform_invert(target_frame), neighbor_frame)
    neighbor_frame_feat = r3.transform2feat(neighbor_frame_relative)

    return torch.cat([
      aa_type,
      same_chain_with_target.unsqueeze(-1),
      is_senpai.unsqueeze(-1),
      rel_pos_to_target,
      neighbor_frame_feat
    ], dim=2)
