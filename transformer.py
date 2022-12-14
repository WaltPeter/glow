import torch
from torch import nn 

from attention import BigBirdBlockSparseAttention
from dense import LightMLP 


class Transformer(nn.Module):
    '''
    Transformer block.
    
    Parameters
    ----------
    dim : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    '''

    def __init__(
            self, 
            batch_size, 
            n_heads, 
            n_patches, 
            device, 
            dim=768, 
            block_size=64, 
            mlp_ratio=4.0, 
            qkv_bias=True, 
            k=8, 
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = BigBirdBlockSparseAttention(
            batch_size=batch_size, 
            seq_length=n_patches, 
            block_size=block_size, 
            hidden_size=dim, 
            num_attention_heads=n_heads, 
            device=device
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = LightMLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
                use_bias=qkv_bias, 
                k=k, 
        )

    def forward(self, x): 
        x2, x1 = x.chunk(2, -1)
        x2, x1 = [i.squeeze(-1) for i in [x2, x1]]
        y2 = x2 + self.attn(self.norm1(x1)) 
        y1 = x1 + self.mlp(self.norm2(y2)) 
        y2, y1 = [i.unsqueeze(-1) for i in [y2, y1]] 
        return torch.cat([y1, y2], -1) 

    def reverse(self, y): 
        y1, y2 = y.chunk(2, -1) 
        y1, y2 = [i.squeeze(-1) for i in [y1, y2]] 
        x1 = y1 - self.mlp(self.norm2(y2)) 
        x2 = y2 - self.attn(self.norm1(x1)) 
        x1, x2 = [i.unsqueeze(-1) for i in [x1, x2]]
        return torch.cat([x2, x1], -1) 

