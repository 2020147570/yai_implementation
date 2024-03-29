import copy
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer, norm):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm
    
    # x -> src, src_mask
    def forward(self, src, src_mask):
        out = src # out = x
        for layer in self.layers:
            out = layer(out, src_mask) # out = layer(out)
        out = self.norm(out)
        return out
