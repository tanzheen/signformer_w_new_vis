import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int=0, max_len: int = 5000): 
        '''
        Positional Encoding with maximum length max_len 
        : param size: model size: 
        : param max_len: maximum length of the sequence
        : param dropout: dropout rate
        '''
        super(PositionalEncoding, self).__init__()

        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, size, 2).float() *
                             -(math.log(10000.0) / size))
        
        pe [: , 0::2] = torch.sin(position * div_term)
        pe [: , 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)


        self.register_buffer('pe', pe)
        self.dim = size

    def forward(self, emb):
        '''
        Forward pass of the Positional Encoding
        : param emb: input tensor (batch_size, seq_len, emb_dim)
        : return: Positional Encoding tensor (batch_size, seq_len, emb_dim)
        '''
        return emb + self.pe[:, :emb.size(1)]
    
class CoPE(nn.Module): 
    def __init__(self, npos_max, head_dim): 
        super().__init__()
        self,npos_max = npos_max
        self.pos_emb = nn.Parameter(
            torch.zeros(1, head_dim, npos_max)
        )

    def forward(self, query, attn_logits): 
        # compute  positions 
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)
        