"""Prepare actual input for neural network models.

This module handles preprocessing of protein structure data into feature tensors
that can be fed into neural network models. It converts raw structural information
into numerical representations that capture sequence and spatial relationships.
"""

import torch
from torch import nn
import torch.nn.functional as func
from einops import rearrange  # For tensor reshaping operations

from pe.common import residue_constants as rc  # Constants for protein residues
from pe.common import r3  # 3D geometry operations

def relative_one_hot(offset, max_relative_feature):
    """Convert relative positions to one-hot encoded features.
    
    Args:
        offset: Tensor of relative position offsets between residues
        max_relative_feature: Maximum distance to consider for relative positions
        
    Returns:
        One-hot encoded tensor representing relative positions
    """
    # Shift and clamp offsets to valid range for one-hot encoding
    clamped = torch.clamp(offset + max_relative_feature,
        min=0, max=2 * max_relative_feature)
    return func.one_hot(
        clamped,
        num_classes=2 * max_relative_feature + 1)


class PreProcess(nn.Module):
    """Preprocessing module that converts raw protein features into model inputs.
    
    This class handles the conversion of protein structure data into numerical
    features that capture:
    - Amino acid types
    - Chain membership
    - Relative positions
    - Spatial relationships
    - Structural frames
    
    Args:
        max_relative_feature: Maximum distance to consider for relative positions
        seq_feat_dim: Dimensionality of sequence features
        pair_feat_dim: Dimensionality of pairwise features
    """

    def __init__(self,
        max_relative_feature=5, seq_feat_dim=128, pair_feat_dim=128):
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.seq_feat_dim = seq_feat_dim
        self.pair_feat_dim = pair_feat_dim

    def forward(self, feat):
        """Process raw features into model-ready inputs.
        
        Args:
            feat: Dictionary containing raw protein features
            
        Returns:
            Tensor containing concatenated features for model input
        """
        # Convert amino acid types to one-hot encoding
        aa_type = func.one_hot(feat['neighbor_name'],
            num_classes=len(rc.restypes) + 1).float()

        # Create feature indicating if neighbor is in same chain as target
        same_chain_with_target = (feat['neighbor_chain'] ==\
            rearrange(feat['target_chain'], 'B -> B ()')).type(torch.float)

        # Calculate relative positions between residues
        neigbor_position_id = feat['neighbor_chain'] + feat['neighbor_id']
        target_position_id = feat['target_chain'] + feat['target_id']
        offset = neigbor_position_id - target_position_id.unsqueeze(-1)
        rel_pos_to_target = relative_one_hot(offset, self.max_relative_feature)

        # Create feature indicating if neighbor comes before target in sequence
        is_senpai = \
            (feat['neighbor_chain'] == feat['target_chain'].unsqueeze(-1)) &\
            (feat['neighbor_id'] < feat['target_id'].unsqueeze(-1))
        is_senpai = is_senpai.type(torch.float)

        # Calculate relative spatial orientations between residues
        neighbor_main_chain_coord = feat['neighbor_main_chain_coord']
        neighbor_frame = r3.vec2transform(neighbor_main_chain_coord)
        target_frame = r3.vec2transform(feat['target_main_chain_coord'])
        neighbor_frame_relative = torch.einsum('Bij,BLjk->BLik',
            r3.transform_invert(target_frame), neighbor_frame)
        neighbor_frame_feat = r3.transform2feat(neighbor_frame_relative)

        # Concatenate all features along the feature dimension
        return torch.cat([
            aa_type,  # Amino acid type features
            same_chain_with_target.unsqueeze(-1),  # Chain membership
            is_senpai.unsqueeze(-1),  # Sequence order
            rel_pos_to_target,  # Relative position
            neighbor_frame_feat  # Spatial orientation
        ], dim=2)
