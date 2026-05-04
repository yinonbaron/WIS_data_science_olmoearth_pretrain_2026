from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat

class PatchifyBandset(nn.Module):
    '''
    A module that takes in a batch of data for a specific bandset and patchifies it using a convolutional layer.
    The convolutional layer has a kernel size and stride equal to the patch size, which allows it to create non-overlapping patches of the input data. 
    The output of the convolutional layer is then rearranged to have the shape (B, H', W', T, D), where B is the batch size, H' and W' are the height 
    and width of the patchified data, T is the number of time steps, and D is the embedding size.
    '''
    def __init__(self, 
                patch_size: int,
                input_channels: int,
                embedding_size: int,
                bias: bool = True,
                ):
        '''
        Args:
            patch_size: the size of the patches to be created
            input_channels: the number of channels in the input data for this bandset
            embedding_size: the size of the output embedding for each patch
            bias: whether to include a bias term in the convolutional layer
        '''

        super().__init__()

        self.patch_size = patch_size
        
        # A 2D convolution layer with stride = kernel_size creates non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=embedding_size, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the PatchifyBandset module.
        Args:
            x: a tensor of shape (B, H, W, T, C) containing the input data for a specific bandset
        Returns:            
        a tensor of shape (B, H', W', T, D) containing the patchified and embedded output, where H' and W' are the height and width of the patchified data
        '''
        
        # x is expected to be shape: (B, H, W, T, C)
        B, H, W, T, C = x.shape
        
        # Conv2d expects inputs of shape (N, C, H, W). We fuse Batch and Time.
        x_rearranged = rearrange(x, 'b h w t c -> (b t) c h w')
        
        # Apply the convolution to extract patches
        # Output shape: (B*T, D, H', W')
        x_proj = self.proj(x_rearranged)
        
        # Rearrange back to the expected target shape (B, H', W', T, D)
        out = rearrange(x_proj, '(b t) d h w -> b h w t d', b=B, t=T)
        
        return out

class PatchEmbeddings(nn.Module):
    '''
    A module that takes in a batch of data and patchifies it using a separate PatchifyBandset module for each bandset in the input data.
    The output of each PatchifyBandset module is then combined into a single tensor containing the patchified and embedded output for all 
    bandsets in the input data, along with the corresponding output mask.
    '''
    def __init__(self, 
                 patch_size: int,
                embedding_size: int,
                modality: str,
                ):
        '''
        Constructor for the PatchEmbeddings module.
        Args:
            patch_size: the size of the patches to be created
            embedding_size: the size of the output embedding for each patch
            modality: the name of the modality for which to create the patch embeddings
        '''

        super().__init__()
        
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.modality = modality

        self.bandset_patchifiers = nn.ModuleList()

        for bandset_indices in BANDSETS[modality]:
            # The number of input channels for this patchifier is the length of the bandset list
            self.bandset_patchifiers.append(PatchifyBandset(
                patch_size=patch_size, 
                input_channels=len(bandset_indices), 
                embedding_size=embedding_size
            ))
            
    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        '''
        Forward pass of the PatchEmbeddings module.
        Args:
            x: a dictionary containing the input data and timestamps for a specific modality, with keys:
                - modality: a tensor of shape (B, H, W, T, C) containing the input data for the modality
                - modality_mask: a tensor of shape (B, H, W, T, C) containing the input mask for the modality
                - timestamps: a tensor of shape (B, T, 3) containing the timestamps for each patch, where the last dimension contains (day, month, year)
            modality: the name of the modality for which to create the patch embeddings
        Returns:
            a dictionary containing the patchified and embedded output for the modality, along with the corresponding output mask and timestamps, with keys:
                - modality: a tensor of shape (B, H', W', T, Bs, D) containing the patchified and embedded output for the modality, where H' and W' are the height and width of the patchified data, Bs is the number of band sets, and D is the embedding size
                - modality_mask: a tensor of shape (B, H', W', T, Bs) containing the output mask for the modality, where H' and W' are the height and width of the patchified data, Bs is the number of band sets, and T is the number of time steps
                - timestamps: a tensor of shape (B, T, 3) containing the timestamps for each patch, where the last dimension contains (day, month, year)
        '''

        data = x[modality]           # (B, H, W, T, C)
        mask = x[f"{modality}_mask"] # (B, H, W, T)
        timestamps = x['timestamps'] # (B, T, 3)
        
        patchified_outputs = []
        
        # Iterate over the patchifiers and the corresponding channel indices simultaneously
        for patchifier, bandset_indices in zip(self.bandset_patchifiers, BANDSETS[modality]):
            # Extract the specific channels for this bandset using advanced indexing.
            # This safely pulls out the exact channels (e.g., [4, 5, 6, 7, 8, 9]) from the last dimension.
            bandset_data = data[..., bandset_indices] 
            
            # Pass the extracted channels through the patchifier and store the result
            patchified_outputs.append(patchifier(bandset_data))
        
        # Combine the patchified bandsets by summing them together
        out_data = torch.stack(patchified_outputs, dim=4)
        
        # Downsample the mask to match the new spatial dimensions H' and W'
        # Stride slicing works perfectly here since stride = patch_size
        out_mask = mask[:, ::self.patch_size, ::self.patch_size, :]
        
        return {
            modality: out_data,           # (B, H', W', T, D)
            f"{modality}_mask": out_mask, # (B, H', W', T)
            "timestamps": timestamps      # (B, T, 3)
        }    

class CompositeEncoding(nn.Module):
    '''
    A module that creates encodings for channel, time, month, and space.

    We have four types of encodings:
        1. Channel encoding: a learnable embedding for each channel in the input data
        2. Time encoding: a fixed sinusoidal encoding based on the timestamps of the input data
        3. Month encoding: a fixed embedding based on the month of the timestamps of the input data
        4. Space encoding: a fixed sinusoidal encoding based on the spatial location of the patches in the input data, scaled by the resolution of the input data.

        Since we have four types of encodings, we divide the embedding dimension by 4 and allocate an equal portion to each encoding type. 
        We then concatenate the four encodings together to get the final output embedding for each patch.

    '''
    def __init__(self, modality: str, embedding_size: int):
        '''
        Constructor for the CompositeEncoding module.
        Args:
            modality: the name of the modality for which to create the encodings
            embedding_size: the size of the output embedding for each patch
        '''
        super().__init__()
        
        self.modality = modality
        self.embedding_size = embedding_size
        
        # Divide embedding dimension by 4 for the 4 encoding types
        self.d_enc = embedding_size // 4
        
        # 1. Channel Encoding: Learnable embedding
        self.num_bandsets = len(BANDSETS[modality])
        self.channel_embed = nn.Embedding(self.num_bandsets, self.d_enc)
        nn.init.zeros_(self.channel_embed.weight)
        
        # 3. Month Encoding: Fixed embedding
        angles = torch.arange(0, 13) / (12 / (2 * 3.141592653589793))
        dim_per_table = self.d_enc // 2
        sin_table = torch.sin(angles.unsqueeze(-1).expand(-1, dim_per_table))
        cos_table = torch.cos(angles.unsqueeze(-1).expand(-1, dim_per_table))
        month_table = torch.cat([sin_table[:-1], cos_table[:-1]], dim=-1)
        self.month_embed = nn.Embedding.from_pretrained(month_table, freeze=True)

    def _get_sinusoidal_encoding(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Helper to create fixed sinusoidal embeddings based on positions."""
        omega = torch.arange(dim // 2, device=positions.device) / dim / 2.0
        inv_freq = 1.0 / (10000 ** omega)
        
        out = positions.unsqueeze(-1) * inv_freq
        sin_enc = torch.sin(out)
        cos_enc = torch.cos(out)
        
        enc = torch.cat([sin_enc, cos_enc], dim=-1)
        return enc

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass of the CompositeEncoding module. 

        Args:
            x: a dictionary containing the input data and timestamps for a specific modality, with keys:
                - modality: a tensor of shape (B, H', W', T, C) containing the patchified data for the modality
                - timestamps: a tensor of shape (B, T, 3) containing the timestamps for each patch, where the last dimension contains (day, month, year)
        Returns:
            a tensor of shape (B, H', W', T, Bs, D) containing the combined encoding for the modality, where Bs, is the number of band sets and D is the embedding size
        '''
        
        patch_data = x[self.modality]  # (B, H', W', T, C/D)
        timestamps = x['timestamps']   # (B, T, 3) -> Last dim is (day, month, year)
        
        B, H_prime, W_prime, T, Bs, D = patch_data.shape
        device = patch_data.device
        
        # 1. Channel Encoding
        c_idx = torch.arange(Bs, device=device)
        c_enc_base = self.channel_embed(c_idx)
        c_enc = c_enc_base.view(1, 1, 1, 1, Bs, self.d_enc).expand(B, H_prime, W_prime, T, Bs, self.d_enc)
        
        # 2. Time Encoding
        t_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        t_enc_base = self._get_sinusoidal_encoding(t_positions, self.d_enc)
        t_enc = t_enc_base.view(B, 1, 1, T, 1, self.d_enc).expand(B, H_prime, W_prime, T, Bs, self.d_enc)
        
        # 3. Month Encoding
        months_idx = timestamps[..., 1].long()
        m_enc_base = self.month_embed(months_idx)
        m_enc = m_enc_base.view(B, 1, 1, T, 1, self.d_enc).expand(B, H_prime, W_prime, T, Bs, self.d_enc)
        
        # 4. Space Encoding
        y_pos = torch.arange(H_prime, device=device) * PATCH_SIZE
        x_pos = torch.arange(W_prime, device=device) * PATCH_SIZE
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        s_dim = self.d_enc // 2
        y_enc = self._get_sinusoidal_encoding(grid_y, s_dim)
        x_enc = self._get_sinusoidal_encoding(grid_x, s_dim)
        
        # Note: concatenate x then y!
        s_enc_base = torch.cat([x_enc, y_enc], dim=-1)
        s_enc = s_enc_base.view(1, H_prime, W_prime, 1, 1, self.d_enc).expand(B, H_prime, W_prime, T, Bs, self.d_enc)
        
        composite = torch.cat([c_enc, t_enc, m_enc, s_enc], dim=-1)
        
        return composite