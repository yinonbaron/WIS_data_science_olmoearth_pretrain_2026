from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "02_patchify_encoding"))
from HW02 import PatchEmbeddings, CompositeEncoding

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
        q = rearrange(self.q_proj(x), 'B N_q (num_heads head_dim) -> B num_heads N_q head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        k = rearrange(self.k_proj(y), 'B N_kv (num_heads head_dim) -> B num_heads N_kv head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        v = rearrange(self.v_proj(y), 'B N_kv (num_heads head_dim) -> B num_heads N_kv head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        
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
        # Should we dropout?
        # 1. First Layer Normalization (applied before Attention)
        self.norm1 = nn.LayerNorm(embedding_size)
        
        # 2. The Multi-Head Attention module we just built
        self.attn = Attention(embedding_size, num_heads)
        
        # 3. Second Layer Normalization (applied before the MLP)
        self.norm2 = nn.LayerNorm(embedding_size)
        
        # 4. The Feed-Forward Network (MLP)
        # We expand the embedding size by the mlp_ratio (typically 4x) to give the network 
        # a larger "hidden workspace" to process the attention results, then project it back down.
        hidden_dim = int(embedding_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.GELU(), # Standard activation function for Transformers
            nn.Linear(hidden_dim, embedding_size)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, attn_mask: torch.Tensor=None) -> torch.Tensor:
        '''
        Forward pass for the AttentionBlock module.

        Args:
            x: The input tensor of shape (batch_size, seq_length, embedding_size) to be used as the query. If y is None, this will also be used as the key and value for self-attention.
            y: An optional input tensor of shape (batch_size, seq_length, embedding_size) to be used as the key and value for cross-attention. 
            attn_mask: An optional attention mask tensor of shape (batch_size, seq_length) where masked positions are indicated by True values. 
            This mask will be applied to the attention weights to prevent attending to certain positions. If None, no masking will be applied.
        '''

        # We normalize 'x' before passing it in as the Query. 
        # We assume y is normelized
        attn_out = self.attn(self.norm1(x), y=y, attn_mask=attn_mask)
        
        # Adding the attention findings back to the original input
        x = x + attn_out
        
       # We normalize the updated 'x' before passing it through the Feed-Forward Network.
        mlp_out = self.mlp(self.norm2(x))
        
        x = x + mlp_out
        
        return x

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

        # Patchification step
        self.patch_embed = PatchEmbeddings(patch_size, embedding_size, modality)

        # The Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        # Final Normalization & Projector
        # Standard in Vision Transformers to normalize before pooling.
        self.norm = nn.LayerNorm(embedding_size)
        
        # This linear layer acts as a final "summary translator" to project the pooled representation.
        self.projector = nn.Linear(embedding_size, embedding_size)

    def forward(self, x: dict[str, torch.Tensor], apply_attn: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the Encoder module.

        Args:
            x: The input dictionary containing the modality data and the corresponding mask and timestamps.
            apply_attn: A boolean flag indicating whether to apply the attention blocks. 
        '''
        
        # --- Patchification ---
        patch_dict = self.patch_embed(x, self.modality)
        patch_data = patch_dict[self.modality] # (B, H', W', T, Bs, D)
        mask = patch_dict[f'{self.modality}_mask'] # (B, H', W', T)

        # --- Add Composite Encodings ---
        encoded_data = patch_data + self.encodings(patch_dict)

        # --- Flattening to Sequence ---
        B, H, W, T, Bs, D = encoded_data.shape
        
        # Flatten the 6D tensor into standard 3D Transformer shape: (B, Seq_Len, D)
        seq_data = rearrange(encoded_data, 'b h w t bs d -> b (h w t bs) d')
        
        # Flatten the mask. Since the mask doesn't have a Bandset (Bs) dimension initially, 
        # we flatten it and then repeat it 'Bs' times so every bandset shares the same physical mask.
        # Does the mask is the same for all bandsets?
        flat_mask = rearrange(mask, 'b h w t bs -> b (h w t bs)')

        # --- Remove Masked Tokens ---
        # making the attn mask matrix to indicate which tokens are masked (True) and which are not (False)
        bool_attn_mask = (flat_mask != MaskValue.ONLINE_ENCODER) # (B, Seq_Len)

        # --- Apply Attention Blocks ---
        x_processed = seq_data
        if apply_attn:
            # we are preforming multi-head attention in a sequential way, passing the output of one block as the input to the next.
            for block in self.blocks:
                x_processed = block(x_processed, attn_mask=bool_attn_mask)

        # --- Add Back the Masked Tokens ---
        x_restored = torch.where(
            bool_attn_mask.unsqueeze(-1), # where to add
            self.mask_token.to(x_processed.dtype), # what to add
            x_processed # target
        )

        # --- Global Pooling & Projection ---
        x_norm = self.norm(x_restored)

        # We make a matrix of the positions of the valid (unmasked) tokens, dimentions: (B, Seq_Len, 1). 
        valid_mask = (~bool_attn_mask).float().unsqueeze(-1)
        
        # We sum all the valid tokens together, and divide by the total number of valid tokens to get the average (Mean Pooling).
        pooled = (x_norm * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-6)
        
        # Project the final single vector.
        projected = self.projector(pooled)

        return {
            self.modality: x_restored,                 # Maps to output[MODALITY]
            f"{self.modality}_mask": flat_mask,        # Maps to output["sentinel2_l2a_mask"]
            "pooled_tokens": projected                 # Maps to output["pooled_tokens"]
        }