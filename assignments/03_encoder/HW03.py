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
        
        #PUT YOUR CODE HERE

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, attn_mask: torch.Tensor=None) -> torch.Tensor:
        '''
        Forward pass for the Attention module.

        Args:
            x: The input tensor of shape (batch_size, seq_length, embedding_size) to be used as the query. If y is None, this will also be used as the key and value for self-attention.
            y: An optional input tensor of shape (batch_size, seq_length, embedding_size) to be used as the key and value for cross-attention. 
            attn_mask: An optional attention mask tensor of shape (batch_size, seq_length) where masked positions are indicated by True values. 
            This mask will be applied to the attention weights to prevent attending to certain positions. If None, no masking will be applied.
        
        '''
        
        #PUT YOUR CODE HERE

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
