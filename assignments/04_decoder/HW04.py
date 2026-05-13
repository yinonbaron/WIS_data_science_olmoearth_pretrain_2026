from copy import deepcopy
from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F

## PUT HERE THE CODE FROM HW03 FOR THE ATTENTION, ATTENTION BLOCK, BASEENCODERDECODER AND ENCODER MODULES

class Decoder(BaseEncoderDecoder):
    '''
    The decoder module. It:
    1. projects from the encoder embedding space to the decoder embedding space
    2. masks the tokens to be decoded with a learnable mask token
    3. adds the encodings
    4. splits the tokens into context tokens and tokens to decode, along with their corresponding masks
    5. applies the attention blocks
    6. combines back the decoded tokens with the context tokens
    7. applies a final projection to the output tokens
    '''

    def __init__(self, modality: str, patch_size: int, embedding_size: int, num_heads: int=0, depth: int=0, mlp_ratio: int=1):
        '''
        Constructor for the Decoder module.

        Args:
            modality: The modality of the input data (e.g., 'sentinel2_l2a').
            patch_size: The size of the patches to be extracted from the input data.
            embedding_size: The dimensionality of the token embeddings.
            num_heads: The number of attention heads to use in the multi-head attention modules within the attention blocks.
            depth: The number of attention blocks to stack in the decoder.
            mlp_ratio: The expansion ratio for the hidden layer in the MLPs within the attention blocks. The hidden layer will have dimensionality embedding_size * mlp_ratio.
        '''
        
        super().__init__(modality, patch_size, embedding_size, num_heads, depth, mlp_ratio)
        # PUT YOUR CODE HERE

            
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        '''
        The forward pass for the Decoder module.
        
        Args:
            x: The input dictionary containing the modality data and the corresponding mask and timestamps.
        
        Returns:
            A dictionary containing the decoded modality data, the corresponding mask, and the timestamps.
        '''
        
        # PUT YOUR CODE HERE
        
class Model(nn.Module):
    '''
    A wrapper class that contains:
    1. the Encoder 
    2. Decoder
    3. The target encoder that shares the same architecture as the encoder but with frozen random weights. 
    The target encoder is used to generate target representations for the masked tokens in the decoder.
    '''

    def __init__(self, encoder_config: dict, decoder_config: dict, modality: str = MODALITY):
        '''
        Constructor for the Model class.

        Args:
            encoder_config: A dictionary containing the configuration parameters for the Encoder module.
            decoder_config: A dictionary containing the configuration parameters for the Decoder module.
            modality: The modality of the input data (e.g., 'sentinel2_l2a').
        '''

        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.modality = modality
        self.mask_name = f"{modality}_mask"

    def model_forward(self, batch):
        '''
        A function that performs the forward pass through the entire model, including the encoder, decoder, and target encoder.
        
        Args:
            batch: A dictionary containing the input data and corresponding masks and timestamps for a batch of samples
        
        Returns:
            A tuple containing the outputs of the encoder, decoder, and target encoder, which are dictionaries containing the modality data,
            corresponding masks, and timestamps.
        '''

        # PUT YOUR CODE HERE