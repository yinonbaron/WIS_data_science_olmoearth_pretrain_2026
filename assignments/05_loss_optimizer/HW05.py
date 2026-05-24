from copy import deepcopy
from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat
from WIS_DSP_lib.dataloader import DataLoader
import torch.nn.functional as F
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "04_decoder"))
from HW04 import Model


## PUT HERE THE CODE FROM ALL PREVIOUS HOMEWORKS FOR THE PATCH EMBEDDINGS, COMPOSITE ENCODING, ATTENTION, ATTENTION BLOCK, BASEENCODERDECODER, ENCODER, DECODER AND MODEL MODULES

class TrainModule():
    """
    A training module that pieces together the model, the dataloader, and the optimizer to perform training steps. It includes:
    1. A method which performs a forward pass through the models, calculates the loss and performs backpropagation for a single batch.
    2. A method which performs an optimization step, including gradient clipping.
    """

    def __init__(self, modality=MODALITY, dataloader: DataLoader = None, optimizer: torch.optim.Optimizer = None, model: Model=None):
        """
        A constructor for the TrainModule class.

        Args:
            modality: The modality of the input data (e.g., 'sentinel2_l2a').
            dataloader: An instance of the DataLoader class that provides batches of training data.
            optimizer: An instance of a PyTorch optimizer that will be used to update the model parameters during training.
            model: An instance of the Model class that contains the encoder, decoder, and target encoder modules to be trained.
        """
        
        self.modality = modality
        self.mask_name = f"{modality}_mask"
        self.model = model
        self.device = model.device
        self.dataloader = dataloader
        self.optimizer = optimizer

    def _move_to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_to_device(val) for key, val in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(val) for val in value)
        if isinstance(value, list):
            return [self._move_to_device(val) for val in value]
        return value
    
    def train_batch(self, batch):
        """
        A method that performs a forward pass through the model, calculates the loss, and performs backpropagation for a single batch of data.

        Args:
            batch: A dictionary containing the input data and corresponding masks and timestamps for a batch of samples
        
        Returns:
            The total loss for the batch, which is the sum of the patch discrimination loss and the NCE loss.
        """
        
        batch_a, batch_b = self._move_to_device(batch)

        latent_a, decoded_a, target_output_a = self.model.model_forward(batch_a)
        latent_b, decoded_b, target_output_b = self.model.model_forward(batch_b)

        tau = 0.1

        # --- Patch discrimination loss ---
        # For each batch: compare the decoder's predicted tokens against the target encoder
        # encoder's tokens at the same DECODER-masked positions, per sample.
        def patch_disc(decoded, target):
            pred = rearrange(decoded[self.modality],   'b h w t bs d -> b (h w t bs) d')
            tgt  = rearrange(target[self.modality],    'b h w t bs d -> b (h w t bs) d')
            mask = rearrange(decoded[self.mask_name],  'b h w t bs -> b (h w t bs)')

            # Concatenate all DECODER tokens across the batch into a single sequence,
            # then split back per sample using per-sample token counts.
            decoder_pos = mask == MaskValue.DECODER
            all_pred = F.normalize(pred[decoder_pos].unsqueeze(0), p=2, dim=-1)  # (1, N_total, D)
            all_tgt  = F.normalize(tgt[decoder_pos].unsqueeze(0),  p=2, dim=-1)  # (1, N_total, D)
            count = decoder_pos.sum(dim=-1)  # (B==num_of_clips,)

            losses, start = [], 0
            for c in count:
                end = start + c
                pred_s = all_pred[:, start:end, :]   # (1, c, D)
                tgt_s  = all_tgt[:,  start:end, :]   # (1, c, D)
                # (1, c, c) similarity matrix; each row i should be most similar to col i
                scores = torch.einsum('npd,nqd->npq', pred_s, tgt_s) / tau
                labels = torch.arange(c, dtype=torch.long, device=pred_s.device)[None]
                loss_s = F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1), reduction='none') * (tau * 2)
                losses.append(loss_s.mean())
                start = end
            return torch.stack(losses).mean()

        patch_loss = (patch_disc(decoded_a, target_output_a) + patch_disc(decoded_b, target_output_b)) / 2

        # --- InfoNCE loss (cross-view) ---
        # The encoder already mean-pools the ONLINE_ENCODER tokens and projects them
        # Push the two views of the same sample together, other samples apart.
        pooled_a = F.normalize(latent_a["pooled_tokens"], p=2, dim=-1)  # (B, D)
        pooled_b = F.normalize(latent_b["pooled_tokens"], p=2, dim=-1)  # (B, D)

        # B×B similarity matrix; diagonal = positive pairs (same sample, different augmentation)
        logits = pooled_a @ pooled_b.T / tau
        labels = torch.arange(pooled_a.shape[0], device=pooled_a.device)
        nce_loss = F.cross_entropy(logits, labels)

        total_loss = patch_loss + nce_loss
        total_loss.backward()
        return total_loss

    def optim_step(self):
        """
        A method that performs an optimization step, including gradient clipping.

        """
        # Clip gradient norms before stepping, matching max_grad_norm=1.0 in the reference.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()