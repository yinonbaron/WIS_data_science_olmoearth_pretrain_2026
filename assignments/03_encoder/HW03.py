from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "02_patchify_encoding"))
from HW02 import PatchEmbeddings, CompositeEncoding
from einops import rearrange, repeat

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
        
        # 2. & 3. Scaled dot-product attention (matches reference F.scaled_dot_product_attention numerics)
        if attn_mask is not None:
            # Student convention: True = masked-out. SDPA convention: True = participate. Invert + expand.
            sdpa_mask = (~attn_mask)[:, None, None].repeat(1, self.num_heads, N_q, 1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        # 4. Re-assemble the heads
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
        # We assume y is normalized - check that
        attn_out = self.attn(self.norm1(x), y=y, attn_mask=attn_mask)
        
        # Adding the attention findings back to the original input
        x = x + attn_out
        
       # We normalize the updated 'x' before passing it through the Feed-Forward Network.
       # Memory unit - outputs the transition that will be added back to the input
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
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([
            AttentionBlock(embedding_size, num_heads, mlp_ratio)
            for _ in range(depth)
            ])
        self.encodings = CompositeEncoding(modality, embedding_size=embedding_size)
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

        # Final Normalization & Projector
        # Standard in Vision Transformers to normalize before pooling.
        self.norm = nn.LayerNorm(embedding_size)
        
        # This linear layer acts as a final "summary translator" to project the pooled representation.
        self.projector = nn.Linear(embedding_size, embedding_size)

        self.apply(self._init_linear_weights)


    def forward(self, x: dict[str, torch.Tensor], apply_attn: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the Encoder module.

        Args:
            x: The input dictionary containing the modality data and the corresponding mask and timestamps.
            apply_attn: A boolean flag indicating whether to apply the attention blocks. 
        '''
        
        # --- 1. Patchification & Encoding ---
        patch_dict = self.patch_embed(x, self.modality)
        patch_data = patch_dict[self.modality]      # Shape: (B, H', W', T, Bs, D)
        mask = patch_dict[f'{self.modality}_mask']  # Shape: (B, H', W', T, Bs)

        # Conditionally add positional/composite encodings
        if apply_attn:
            patch_data = patch_data + self.encodings(patch_dict)

        # --- 2. Flattening into Sequence Space ---
        # This happens in both paths, so we do it once here.
        B, H, W, T, Bs, D = patch_data.shape
        seq_data = rearrange(patch_data, 'b h w t bs d -> b (h w t bs) d')
        flat_mask = rearrange(mask, 'b h w t bs -> b (h w t bs)')
        
        # Generate raw True/False tracking mask: True = KEEP (online token)
        keep_boolean_mask = (flat_mask == MaskValue.ONLINE_ENCODER) # (B, Seq_Len)
        
        # Baseline state for x_restored
        x_restored = seq_data

        if apply_attn:
            # --- 3. Dynamic Masked Token Removal ---
            # Sort mask descending: True values (1s) cluster at front, False values (0s) at back.
            sorted_keep_mask, sorting_indices = torch.sort(keep_boolean_mask.int(), dim=1, descending=True, stable=True)
            
            # Shift data tokens into identical alignment using sorted index references
            x_processed = seq_data.gather(1, sorting_indices.unsqueeze(-1).expand_as(seq_data))
            
            # Dynamically truncate length to only cover the maximum amount of real tokens seen across the batch slice
            sequence_lengths = sorted_keep_mask.sum(dim=-1)
            max_active_length = sequence_lengths.max().item()
            
            x_processed = x_processed[:, :max_active_length]
            truncated_keep_mask = sorted_keep_mask[:, :max_active_length].bool()
            
            # Invert mask tracking to align with standard block expectations (True = Masked/Dropped)
            block_attn_mask = ~truncated_keep_mask

            # --- 4. Apply Attention Blocks Over Compressed Tensor ---
            for block in self.blocks:
                x_processed = block(x_processed, attn_mask=block_attn_mask)
            
            # Apply LayerNorm strictly to visible valid tokens before filling empty slots
            x_processed = self.norm(x_processed)

            # --- 5. Reconstruction Step (Restore Sequence Order) ---
            # Allocate an empty zero-matrix structure on the correct device
            restored_tensor = torch.zeros(
                (B, flat_mask.shape[1], D), 
                device=x_processed.device, 
                dtype=x_processed.dtype
            )
            
            # Convert our tracking index back to a boolean mask over the uncompressed flat shape
            uncompressed_fill_mask = torch.zeros((B, flat_mask.shape[1]), dtype=torch.bool, device=seq_data.device)
            uncompressed_fill_mask[:, :max_active_length] = truncated_keep_mask
            
            # Step A: Map active tokens into temporary packed positions
            restored_tensor[uncompressed_fill_mask] = x_processed[truncated_keep_mask]
            
            # Step B: Scatter everything back into its precise starting position index layout
            x_restored = restored_tensor.new_zeros(restored_tensor.shape)
            x_restored = x_restored.scatter(1, sorting_indices.unsqueeze(-1).expand_as(restored_tensor), restored_tensor)

        # --- 6. Global Pooling & Projection ---
        # Calculate pool denominator from valid unmasked tokens matrix counts
        valid_counts = keep_boolean_mask.float().sum(dim=1, keepdim=True) # (B, 1)
        
        # Zero out any non-data tracking slots before running sum pooling
        clean_data_mask = keep_boolean_mask.float().unsqueeze(-1)
        pooled = (x_restored * clean_data_mask).sum(dim=1) / torch.clamp(valid_counts, min=1.0)
        projected = self.projector(pooled)

        # --- 7. Unflatten back to original shape ---
        # Reshape (B, Seq_Len, D) back to (B, H, W, T, Bs, D)
        x_restored = rearrange(x_restored, 'b (h w t bs) d -> b h w t bs d', h=H, w=W, t=T, bs=Bs)

        return {
            self.modality: x_restored,                 # Maps to output[MODALITY]
            f"{self.modality}_mask": mask,             # Maps to output["sentinel2_l2a_mask"]
            "pooled_tokens": projected                 # Maps to output["pooled_tokens"]
         }