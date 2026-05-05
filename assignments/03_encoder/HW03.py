from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

# ADD CODE FOR PATCH EMBEDDINGS AND COMPOSITE ENCODING FROM HW02 HERE

class Attention(nn.Module):
    '''
    Multi-head attention module that supports separate key/value inputs for self or cross-attention.
    '''
    def __init__(self, embedding_size: int, num_heads: int):
        '''
        Constructor for Attention module.

        Args:
            embedding_size: The dimensionality of the input and output embeddings.
            num_heads: The number of attention heads to use.        
        '''
        super().__init__()
        
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.head_dim = embedding_size // num_heads

        # Scaling factor: standard in transformers to prevent dot products from getting too large
        self.scale = self.head_dim ** -0.5

        # Projections:
        # 1. Query projection (always comes from the primary input 'x')
        self.q_proj = nn.Linear(embedding_size, embedding_size)
        
        # 2. Key and Value projections 
        # (might come from 'x' for self-attention, or 'y' for cross-attention)
        self.k_proj = nn.Linear(embedding_size, embedding_size)
        self.v_proj = nn.Linear(embedding_size, embedding_size)
        
        # 3. Final output projection to mix the heads back together
        self.out_proj = nn.Linear(embedding_size, embedding_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, attn_mask: torch.Tensor=None) -> torch.Tensor:
        '''
        Forward pass for the Attention module.

        Args:
            x: The input tensor of shape (batch_size, seq_length, embedding_size) to be used as the query. If y is None, this will also be used as the key and value for self-attention.
            y: An optional input tensor of shape (batch_size, seq_length, embedding_size) to be used as the key and value for cross-attention. 
            attn_mask: An optional attention mask tensor of shape (batch_size, seq_length) where masked positions are indicated by True values. 
            This mask will be applied to the attention weights to prevent attending to certain positions. If None, no masking will be applied.
        
        '''
        # x shape: (B, Seq_Len_Q, D), B = number of images in batch, Seq_Len_Q = num of patches in the image sequence (H'*W'*T*Bs), D = embedding dimension
        B, N_q, D = x.shape
        
        # If y is not provided, this is Self-Attention. y becomes x.
        if y is None:
            y = x
            
        # y shape: (B, Seq_Len_KV, D), Seq_Len_KV = num of patches in the sequence
        _, N_kv, _ = y.shape
        
        # 1. Project to get Queries, Keys, and Values
        # We immediately reshape them to separate the heads: (B, Seq_Len, num_heads, head_dim)
        # Then we transpose dimensions 1 and 2 so the sequence length is the innermost dimension for matrix multiplication
        q = rearrange(self.q(x), 'B N_q (num_heads head_dim) -> B num_heads N_q head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        k = rearrange(self.k(y), 'B N_kv (num_heads head_dim) -> B num_heads N_kv head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        v = rearrange(self.v(y), 'B N_kv (num_heads head_dim) -> B num_heads N_kv head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        
        # 2. Calculate Attention Scores (Q * K^T) -> B num_heads N_q head_dim @ B num_heads head_dim N_kv => (B, num_heads, N_q, N_kv), 4D multiplication mult the last two dimensions.
        # This is like a 2D matrix of 2D matrices: for each sample in the batch, for each head, we have a (N_q, head_dim) query matrix multiplied by a (head_dim, N_kv) key matrix, resulting in a (N_q, N_kv) score matrix for each head.
        # This way we concat the heads for the same token? is it what we want to do?
        # We use torch.matmul (@) to multiply the queries by the keys. 
        # k.transpose(-2, -1) flips the last two dimensions of K so the shapes align for dot product.
        scores = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N_q, N_kv)

        # 3. Apply the Mask (Optional)
        if attn_mask is not None:
            # attn_mask is a True/False matrix where True indicates positions that should be masked (not attended to).
            # attn_mask is usually (B, sequence length). We need it to broadcast to (B, num_heads, N_q, N_kv).
            # We replace True (masked) values with a massive negative number (-1e9).
            # When passed through softmax, e^(-1e9) becomes 0, so the network completely ignores those patches.
            # The mask is on the y patches (N_kv)
            scores = scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-1e9'))
            
        # 4. Convert scores to probabilities (Softmax)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Multiply by Values
        # We multiply our probability weights by the actual information (V)
        out = attn_weights @ v # Shape: (B, num_heads, N_q, head_dim)
        
        # 6. Re-assemble the heads
        # Transpose back to (B, N_q, num_heads, head_dim) and flatten the last two dimensions to get back to (B, N_q, D)
        # contiguous() is used to ensure the tensor is stored in memory transposed (1,2), which is necessary for the view operation to work correctly.
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        
        # 7. Final linear projection
        return self.out_proj(out)
        

class AttentionBlock(nn.Module):
    '''
    A module that couples multi-head attention with a feedforward MLP and layer normalization, following the standard transformer architecture.
    '''

    def __init__(self, embedding_size: int, num_heads: int, mlp_ratio: int):
        '''
        Constructor for the AttentionBlock module.

        Args:
            embedding_size: The dimensionality of the input and output embeddings.
            num_heads: The number of attention heads to use in the multi-head attention module.
            mlp_ratio: The expansion ratio for the hidden layer in the MLP. The hidden layer will have dimensionality embedding_size * mlp_ratio.
        '''

        super().__init__()
        
        #PUT YOUR CODE HERE

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, attn_mask: torch.Tensor=None) -> torch.Tensor:
        '''
        Forward pass for the AttentionBlock module.

        Args:
            x: The input tensor of shape (batch_size, seq_length, embedding_size) to be used as the query. If y is None, this will also be used as the key and value for self-attention.
            y: An optional input tensor of shape (batch_size, seq_length, embedding_size) to be used as the key and value for cross-attention. 
            attn_mask: An optional attention mask tensor of shape (batch_size, seq_length) where masked positions are indicated by True values. 
            This mask will be applied to the attention weights to prevent attending to certain positions. If None, no masking will be applied.
        '''

        #PUT YOUR CODE HERE

class BaseEncoderDecoder(nn.Module):
    '''
    A base class for the Encoder and Decoder that defines common components and methods. 
    This includes the composite encoding module and and the attention blocks.
    '''

    def __init__(self, modality: str, patch_size: int, embedding_size: int, num_heads: int=0, depth: int=0, mlp_ratio: int=1):
        super().__init__()
        self.modality = modality
        self.blocks = nn.ModuleList([
            AttentionBlock(embedding_size, num_heads, mlp_ratio)
            for _ in range(depth)
            ])
        self.encodings = CompositeEncoding(modality, embedding_size=128)
        self.apply(self._init_linear_weights)
    
    @staticmethod
    def _init_linear_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class Encoder(BaseEncoderDecoder):
    '''
    The encoder module. It:
    1. performs the pathcification 
    2. adds the composite encodings 
    3. removes the maksed tokens
    4. applies the attention blocks
    5. adds back the masked tokens in their original positions
    6. pools the tokens to get a global representation and projects it.
    '''
    def __init__(self, modality: str, patch_size: int, embedding_size: int, num_heads: int=0, depth: int=0, mlp_ratio: int=1):
        '''
        Constructor for the Encoder module.

        Args:
            modality: The modality of the input data (e.g., 'sentinel2_l2a').
            patch_size: The size of the patches to be extracted from the input data.
            embedding_size: The dimensionality of the token embeddings.
            num_heads: The number of attention heads to use in the multi-head attention modules within the attention blocks.
            depth: The number of attention blocks to stack in the encoder.
            mlp_ratio: The expansion ratio for the hidden layer in the MLPs within the attention blocks. 
            The hidden layer will have dimensionality embedding_size * mlp_ratio.
        '''
        super().__init__(modality, patch_size, embedding_size, num_heads, depth, mlp_ratio)

        #PUT YOUR CODE HERE


    def forward(self, x: dict[str, torch.Tensor], apply_attn: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the Encoder module.

        Args:
            x: The input dictionary containing the modality data and the corresponding mask and timestamps.
            apply_attn: A boolean flag indicating whether to apply the attention blocks. 
        '''
        
        #PUT YOUR CODE HERE
