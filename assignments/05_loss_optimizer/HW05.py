from copy import deepcopy
from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat
from WIS_DSP_lib.patchify import PatchEmbeddings, CompositeEncoding
from WIS_DSP_lib.dataloader import DataLoader
import torch.nn.functional as F

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

        # PUT YOUR CODE HERE FOR THE FORWARD PASS, LOSS CALCULATION AND BACKPROPAGATION

    def optim_step(self):
        """
        A method that performs an optimization step, including gradient clipping.
    
        """
        
        # PUT YOUR CODE HERE FOR THE OPTIMIZATION STEP, INCLUDING GRADIENT CLIPPING