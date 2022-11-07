"""Modules and code used in the core part of PE"""
from torch import nn

class Transformer(nn.Module):

  def __init__(self, d_input, d_output, d_model=256, nhead=4, nlayer=3, **kw):
    super().__init__()

    layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, **kw)
    self.input = nn.Linear(d_input, d_model, **kw)
    self.main = nn.TransformerEncoder(layer, num_layers=nlayer)
    self.output = nn.Linear(d_model, d_output, **kw)

  def forward(self, feat, mask):
    """(B, C) <- (B, L, D), (B, L)"""
    feat = self.input(feat)
    logits = self.main(feat, src_key_padding_mask=~mask)
    logits = logits.mean(dim=1)
    return self.output(logits)
