from copy import deepcopy
from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "02_patchify_encoding"))
from HW02 import PatchEmbeddings, CompositeEncoding
sys.path.append(str(Path(__file__).resolve().parents[1] / "03_encoder"))
from HW03 import Attention, AttentionBlock, BaseEncoderDecoder, Encoder


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

    def __init__(self, modality: str, patch_size: int, embedding_size: int, num_heads: int = 0, depth: int = 0, mlp_ratio: int = 1):
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
        self.project_encoder_to_decoder = nn.Linear(embedding_size, embedding_size, bias=True)
        self.to_output_embed = nn.Linear(embedding_size, embedding_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(embedding_size))
        self.input_norm = nn.LayerNorm(embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.apply(self._init_linear_weights)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        '''
        The forward pass for the Decoder module.

        Args:
            x: The input dictionary containing the modality data and the corresponding mask and timestamps.

        Returns:
            A dictionary containing the decoded modality data, the corresponding mask, and the timestamps.
        '''
        data = x[self.modality]             # (B, H', W', T, Bs, D)
        mask = x[f"{self.modality}_mask"]   # (B, H', W', T, Bs)
        timestamps = x["timestamps"]         # (B, T, 3)

        # --- 1. Normalize + project encoder tokens into decoder space ---
        data = self.project_encoder_to_decoder(self.input_norm(data))

        # --- 2. Replace DECODER positions with learnable mask token ---
        decoder_positions = (mask == MaskValue.DECODER).unsqueeze(-1)  # (B, H', W', T, Bs, 1)
        mask_token_expanded = self.mask_token.view(1, 1, 1, 1, 1, -1).expand_as(data)
        data = torch.where(decoder_positions, mask_token_expanded, data)

        # --- 3. Add composite encodings ---
        # enc_dict is like x after we change the data
        enc_dict = {self.modality: data, f"{self.modality}_mask": mask, "timestamps": timestamps}
        data = data + self.encodings(enc_dict)

        # --- 4. Flatten ---
        B, H, W, T, Bs, D = data.shape
        seq_data = rearrange(data, 'b h w t bs d -> b (h w t bs) d')
        flat_mask = rearrange(mask, 'b h w t bs -> b (h w t bs)')
        T_seq = flat_mask.shape[1]

        # --- 5. Sort: DECODER(2) → front, TARGET_ENCODER_ONLY(1) → middle, ONLINE_ENCODER(0) → back ---
        sort_mask = flat_mask.clone().int()
        sort_mask[sort_mask == MaskValue.MISSING] = MaskValue.TARGET_ENCODER_ONLY # maybe uneccesery?
        sorted_mask, indices = torch.sort(sort_mask, dim=1, descending=True, stable=True)
        seq_data = seq_data.gather(1, indices.unsqueeze(-1).expand_as(seq_data))

        is_decode = sorted_mask == MaskValue.DECODER          # True = real DECODER token
        is_context = sorted_mask == MaskValue.ONLINE_ENCODER  # True = real context token

        max_decode_len = int(is_decode.sum(dim=-1).max().item())
        tokens_to_decode = seq_data[:, :max_decode_len]           # (B, X_len, D)
        decode_mask = is_decode[:, :max_decode_len]               # True = real token

        # Take context from the back; guard against -0 == 0 slice edge case
        max_context_len = int(is_context.sum(dim=-1).max().item())
        context_tokens = seq_data[:, T_seq - max_context_len:]    # (B, Y_len, D)
        context_mask = is_context[:, T_seq - max_context_len:]    # True = real token

        # --- 6. Cross-attention blocks: Q = tokens_to_decode, K/V = context_tokens ---
        cross_attn_mask = ~context_mask
        for block in self.blocks:
            tokens_to_decode = block(tokens_to_decode, y=context_tokens, attn_mask=cross_attn_mask)

        # --- 7. Combine back (mirrors reference combine_x_y) ---
        combined = torch.zeros(B, T_seq, D, device=seq_data.device, dtype=seq_data.dtype)
        combined[:, T_seq - max_context_len:] = context_tokens * context_mask.unsqueeze(-1)
        combined[:, :max_decode_len] = tokens_to_decode * decode_mask.unsqueeze(-1)
        # Scatter back to original positions (inverse of the sort+gather above)
        combined = combined.scatter(1, indices.unsqueeze(-1).expand_as(combined), combined)

        # --- 8. Unflatten, then apply norm + output projection per bandset ---
        combined = rearrange(combined, 'b (h w t bs) d -> b h w t bs d', h=H, w=W, t=T, bs=Bs)
        output_bandsets = []
        for bs_idx in range(Bs):
            tokens_bs = combined[:, :, :, :, bs_idx, :]  # (B, H', W', T, D)
            output_bandsets.append(self.to_output_embed(self.norm(tokens_bs)))
        output = torch.stack(output_bandsets, dim=4)      # (B, H', W', T, Bs, D)

        return {
            self.modality: output,
            f"{self.modality}_mask": mask,
            "timestamps": timestamps,
        }


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
        # 1. Encode the online (masked) batch
        latent = self.encoder(batch)
        latent["timestamps"] = batch["timestamps"]

        # 2. Decode: cross-attention from DECODER tokens attending to ONLINE_ENCODER context
        decoded = self.decoder(latent)

        # 3. Target encoder: all non-MISSING tokens visible, no encodings, no attention blocks
        #    Mirrors reference token_exit_cfg={MODALITY: 0} which returns norm(patchified) tokens
        mask = batch[self.mask_name]
        unmasked_mask = torch.where(mask == MaskValue.TARGET_ENCODER_ONLY, mask, torch.zeros_like(mask))
        unmasked_batch = {
            self.modality: batch[self.modality],
            self.mask_name: unmasked_mask,
            "timestamps": batch["timestamps"],
        }
        with torch.no_grad():
            target_output = self.target_encoder(unmasked_batch)

        return latent, decoded, target_output
