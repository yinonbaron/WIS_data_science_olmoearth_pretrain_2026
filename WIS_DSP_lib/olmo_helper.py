from json import encoder
import random
import functools

from olmoearth_pretrain.data.dataset import OlmoEarthDataset
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoader
from olmoearth_pretrain.data.collate import collate_double_masked_batched
from olmoearth_pretrain.train.masking import ModalityCrossRandomMaskingStrategy
from olmoearth_pretrain.data.constants import Modality, BASE_GSD, MAX_SEQUENCE_LENGTH
from olmoearth_pretrain.nn.flexi_vit import Encoder, Predictor
from upath import UPath
import numpy as np
import torch
from upath import UPath
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import ContrastiveLatentMIMTrainModuleConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.trainer import Trainer
from olmoearth_pretrain.train.masking import MaskingConfig
from pathlib import Path
from olmo_core.utils import get_default_device

#%% set global variables

DATA_DIR = 'data/dataset/6/'
seed = 3622
device = get_default_device()

def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
modality = 'sentinel2_l2a'
modality_spec = Modality.get(modality)

set_reproducible_seeds(seed)

def get_dataloader(device=device):
    #%% Build the dataset
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(DATA_DIR),
        dtype=np.float32,
        training_modalities=['sentinel2_l2a'],
        normalize=True,
    )

    num_samples = int(dataset.h5py_dir.name)
    dataset.sample_indices = np.arange(num_samples)
    #%% Build the dataloader

    # define the masking strategy 
    masking_strategy = ModalityCrossRandomMaskingStrategy(
        allow_encoding_decoding_same_bandset=True,
        only_decode_modalities=[
                    Modality.WORLDCOVER.name,
                    Modality.SRTM.name,
                    Modality.OPENSTREETMAP_RASTER.name,
                    Modality.WRI_CANOPY_HEIGHT_MAP.name,
                    Modality.CDL.name,
                    Modality.WORLDCEREAL.name
        ]
    )
    masking_strategy_b = None

    # define the collator with the masking strategy
    collator = functools.partial(
        collate_double_masked_batched,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )

    # build the dataloader
    data_loader = OlmoEarthDataLoader(
                dataset=dataset,
                work_dir='./local_output/checkpoints/anonymous/test_run',
                global_batch_size=3,
                dp_world_size=1,
                dp_rank=0,
                fs_local_rank=0,
                seed=3622,
                shuffle=False,
                num_workers=0,
                target_device_type=device.type,
                collator=collator,
                drop_last=True,
                min_patch_size=8,
                max_patch_size=8,
                sampled_hw_p_list=[8],
                token_budget=2250,
                num_dataset_repeats_per_epoch=1,
                transform=None,
                masking_strategy=masking_strategy,
                masking_strategy_b=None,
                num_masked_views=2,
                tokenization_config=None,
            )
    
    # set the dataloader to the first epoch and build the global indices
    data_loader._global_indices = data_loader._build_global_indices()
    data_loader._epoch = 1

    return data_loader, dataset

def get_encoder():
    encoder = Encoder(
        embedding_size=128,
        num_heads=8,
        depth=4,
        mlp_ratio=4,
        supported_modalities=[modality_spec],
        max_patch_size=8,
        min_patch_size=8,
        drop_path=0.,
        max_sequence_length=12,
        num_register_tokens=0,
        learnable_channel_embeddings=True,
        random_channel_embeddings=False,
        num_projection_layers=1,
        aggregate_then_project=True,
        use_flash_attn=False,
        frozen_patch_embeddings=False,
        qk_norm=False,
        log_token_norm_stats=False,
        tokenization_config=None,
        band_dropout_rate=0.0,
        random_band_dropout=False,
        band_dropout_modalities=None,
    )
    return encoder

def get_decoder():
    decoder = Predictor(
        encoder_embedding_size=128,
        decoder_embedding_size=128,
        num_heads=8,
        depth=4,
        mlp_ratio=4,
        supported_modalities=[modality_spec],
        drop_path=0,
        max_sequence_length=12,
        learnable_channel_embeddings=True,
        random_channel_embeddings=False,
        use_flash_attn=False,
        qk_norm=False,
        tokenization_config=None,
    )
    return decoder

#%% Build a trainer for the model

def get_trainer(model, device, data_loader):
    
    masking_config = MaskingConfig(
        strategy_config={
            "type": "modality_cross_random",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "allow_encoding_decoding_same_bandset": True,
            "only_decode_modalities": [
                Modality.WORLDCOVER.name,
                Modality.SRTM.name,
                Modality.OPENSTREETMAP_RASTER.name,
                Modality.WRI_CANOPY_HEIGHT_MAP.name,
                Modality.CDL.name,
                Modality.WORLDCEREAL.name,
            ],
        },
        tokenization_config=None,
    )

    train_module_config = ContrastiveLatentMIMTrainModuleConfig(
            optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
            rank_microbatch_size=3,
            masking_config=masking_config,
            loss_config=LossConfig(
                loss_config={
                    "type": "modality_patch_discrimination_new",
                    "tau": 0.1,
                }
            ),
            contrastive_config=LossConfig(
                loss_config={
                    "type": "InfoNCE",
                    "weight": 0.1,
                }
            ),
            token_exit_cfg={modality: 0 for modality in [modality]},
            max_grad_norm=1.0,
            scheduler=None,#CosWithWarmup(warmup_steps=8000),
            ema_decay=(1.0, 1.0),
        )

    train_module = train_module_config.build(model, device=device)
    trainer = Trainer(
                train_module=train_module,
                data_loader=data_loader,
                checkpointer=None,
                work_dir=Path("."),
                device=torch.device(device),
                dp_process_group=None,
                callbacks={},
                save_folder=Path("."),
                max_duration=100
            )
    return trainer