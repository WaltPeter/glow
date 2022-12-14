import math 
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


class BigBirdBlockSparseAttention(nn.Module): 

    def __init__(self, batch_size, seq_length, block_size, num_attention_heads, hidden_size, device):
        super().__init__()

        past_key_values_length = 0
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device) 

        self.blocked_encoder_mask, self.band_mask, self.from_mask, self.to_mask = self.create_masks_for_block_sparse_attn(
            attention_mask, block_size
        )

        self.block_size = block_size 
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = num_attention_heads * self.attention_head_size 

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=True)

    
    def forward(self, hidden_states): 
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        assert from_seq_length % from_block_size == 0, "Query sided sequence length must be multiple of block size"
        assert to_seq_length % to_block_size == 0, "Key/Value sided sequence length must be multiple of block size"

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        context_layer = self.bigbird_block_sparse_attention(
                            query_layer=query_layer, 
                            key_layer=key_layer, 
                            value_layer=value_layer, 
                            band_mask=self.band_mask, 
                            from_mask=self.from_mask, 
                            to_mask=self.to_mask, 
                            num_attention_heads=self.num_attention_heads, 
                            attention_head_size=self.attention_head_size, 
                            from_block_size=from_block_size, 
                            to_block_size=to_block_size, 
                            batch_size=batch_size, 
                            from_seq_length=from_seq_length, 
                            to_seq_length=to_seq_length, 
                        )
        
        return context_layer 


    # @staticmethod
    def torch_bmm_nd(self, inp_1, inp_2, ndim=None):
        """ Fast nd matrix multiplication """
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    # @staticmethod
    def torch_bmm_nd_transpose(self, inp_1, inp_2, ndim=None):
        """ Fast nd matrix multiplication with transpose """
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))


    # @staticmethod
    def create_masks_for_block_sparse_attn(self, attention_mask: torch.Tensor, block_size: int):

        batch_size, seq_length = attention_mask.size()
        assert (
            seq_length % block_size == 0
        ), "Sequence length must be multiple of block size, but sequence length is {}, while block size is {}.".format(seq_length, block_size)

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.
            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].
            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            exp_blocked_to_pad = torch.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
            )
            band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask

        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask


    # @staticmethod
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def bigbird_block_sparse_attention(
        self, 
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        num_attention_heads,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_length,
        to_seq_length
    ):

        # BigBird block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # ETC:
        #     global tokens: extra_globals_tokens + 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        assert (
            from_seq_length // from_block_size == to_seq_length // to_block_size
        ), "Error the number of blocks needs to be same!"

        # Define shorthands
        h = num_attention_heads
        rsqrt_d = 1 / math.sqrt(attention_head_size)
        b = batch_size
        m = from_seq_length
        n = to_seq_length
        wm = from_block_size
        wn = to_block_size

        blocked_query_matrix = query_layer.view(b, h, m // wm, wm, -1)
        blocked_key_matrix = key_layer.view(b, h, n // wn, wn, -1)
        blocked_value_matrix = value_layer.view(b, h, n // wn, wn, -1)

        # 1st block is global q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * -10000.0
        first_attn_weights = F.softmax(first_product, dim=-1)  # [b, h, wm, n]

        # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)

        # q[1] x (sliding_keys, random_keys, global_keys)

        second_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
            ],
            dim=2,
        )  # [b, h, 4*wn, -1]
        second_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
            ],
            dim=2,
        )  # [b, h, 4*wn, -1]

        # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat(
            [
                to_mask[:, :, :, : 3 * wn],
                to_mask[:, :, :, -wn:],
            ],
            dim=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - second_seq_pad) * -10000.0
        second_attn_weights = F.softmax(second_product, dim=-1)  # [b , h, wm, 4*wn]

        # [b, h, wm, 4*wn] x [b, h, 4*wn, -1] ==> [b, h, wm, -1]
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

        second_context_layer.unsqueeze_(2)

        # q[-2:2] x (sliding_keys, random_keys, global_keys)

        # initialize q,k,v->q,k,v[-2:2]
        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3
        )  # [b, h, m//wm-4, 3*wn, -1]
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            dim=3,
        )  # [b, h, m//wm-4, 3*wn, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]

        # sliding attention scores for q[-2:2]
        # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        #     ==> [b, h, m//wm-4, wm, 3*wn]
        inner_band_product = inner_band_product * rsqrt_d

        # 1st block is global
        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
        first_band_product = first_band_product * rsqrt_d

        # last block is global
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
        last_band_product = last_band_product * rsqrt_d

        # masking padded tokens
        inner_band_product += (1.0 - band_mask) * -10000.0
        first_band_product += (1.0 - to_mask[:, :, :, :wn].unsqueeze(3)) * -10000.0
        last_band_product += (1.0 - to_mask[:, :, :, -wn:].unsqueeze(3)) * -10000.0

        # completing attention scores matrix for all q[-2:2]
        band_product = torch.cat(
            [first_band_product, inner_band_product, last_band_product], dim=-1
        )  # [b, h, m//wm-4, wm, 5*wn]

        # safely doing softmax since attention matrix is completed
        attn_weights = F.softmax(band_product, dim=-1)  # [b, h, m//wm-4, wm, 5*wn]

        # contibution of sliding keys
        # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
        context_layer = self.torch_bmm_nd(attn_weights[:, :, :, :, wn : 4 * wn], exp_blocked_value_matrix, ndim=5)
        #     ==> [b, h, m//wm-4, wm, -1]

        # adding contribution of global keys
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :wn], blocked_value_matrix[:, :, 0]
        )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -wn:], blocked_value_matrix[:, :, -1]
        )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]

        # q[-2] x (sliding_keys, random_keys, global_keys)

        second_last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
            ],
            dim=2,
        )  # [b, h, (4+r)*wn, -1]
        second_last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
            ],
            dim=2,
        )  # [b, h, (4+r)*wn, -1]

        # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat(
            [
                to_mask[:, :, :, :wn],
                to_mask[:, :, :, -3 * wn :],
            ],
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - second_last_seq_pad) * -10000.0
        second_last_attn_weights = F.softmax(second_last_product, dim=-1)  # [b, h, wm, 4*wn]

        # [b, h, wm, 4*wn] x [b, h, 4*wn, -1] ==> [b, h, wm, -1]
        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer.unsqueeze_(2)

        # last block is global q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * -10000.0
        last_attn_weights = F.softmax(last_product, dim=-1)  # [b, h, wm, n]

        # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)
        context_layer = torch.cat(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
            dim=2,
        )
        context_layer = context_layer.view((b, h, m, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)

        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        return context_layer 


class SelfAttention(nn.Module): 
    '''Attention mechanism.
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    '''

    def __init__(self, dim, n_heads=12, qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads 
        self.dim = dim 
        self.head_dim = dim // n_heads 
        self.scale = self.head_dim ** -0.5 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
        q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)

        return x


# batch_size = 2  
# seq_length = 1024 
# block_size = 64 #default 
# hidden_size = 768 
# num_attention_heads = 12 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu") 

# #Input 
# hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)

# att = BigBirdBlockSparseAttention(
#             batch_size=batch_size, 
#             seq_length=seq_length, 
#             block_size=block_size, 
#             hidden_size=hidden_size, 
#             num_attention_heads=num_attention_heads, 
#             device=device
#         ).to(device) 