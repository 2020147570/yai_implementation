import copy
import torch.nn as nn

from ..layer.residual_connection_layer import ResidualConnectionLayer

class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer(copy.deepcopy(norm, dr_rate)) for _ in range(2)]
    
    # x -> src, src_mask
    def forward(self, src, src_mask):
        out = src # out = x
        out = self.residuals[0](out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask)) # out = self.self_attention(out)
        out = self.residuals[1](out, self.position_ff)
        return out
