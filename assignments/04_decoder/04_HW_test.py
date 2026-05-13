from typing_extensions import runtime

from WIS_DSP_lib.dataloader import Dataset, DataLoader
from HW04 import Model
from WIS_DSP_lib.constants import *
from WIS_DSP_lib.olmo_helper import get_dataloader, get_encoder, get_decoder
from WIS_DSP_lib.test_helpers import assert_tensor_values_equal
from olmoearth_pretrain.nn.latent_mim import LatentMIM

seed = 3622

#%% Run the olmoearth_pretrain code 
set_reproducible_seeds(seed)

# load batch
data_loader, dataset = get_dataloader()

data_iterator = iter(data_loader)
batch = next(data_iterator)
encoder = get_encoder()
decoder = get_decoder()
model = LatentMIM(encoder, decoder)

latent_olmo, decoded_olmo, latent_projected_and_pooled_olmo, _, _ = model(batch[1], patch_size=8)
with torch.no_grad():
    target_encoder_output_olmo = model.target_encoder.forward(
        batch[1].unmask(),
        patch_size=8,
        token_exit_cfg={MODALITY: 0},
    )

#%% Run our refactored code

set_reproducible_seeds(seed)
dataset = Dataset(h5py_dir=DATA_DIR)
data_loader = DataLoader(
        dataset=dataset,
        batch_size=GLOBAL_BATCH_SIZE,
    )

batch = next(iter(data_loader))

encoder_config = {
    "modality": MODALITY,
    "embedding_size": 128,
    "patch_size": 8,
    "num_heads": 8,
    "depth": 4,
    "mlp_ratio": 4.0,
}
decoder_config = {
    "modality": MODALITY,
    "embedding_size": 128,
    "patch_size": 8,
    "num_heads": 8,
    "depth": 4,
    "mlp_ratio": 4.0,
}
model = Model(encoder_config, decoder_config)
latent, decoded, target_output = model.model_forward(batch[0])

assert_tensor_values_equal(
    decoded_olmo.sentinel2_l2a,
    decoded["sentinel2_l2a"],
    "tokens",
)

print("Decoded tokens match!")
assert_tensor_values_equal(
    decoded_olmo.sentinel2_l2a_mask,
    decoded["sentinel2_l2a_mask"],
    "masks",
)
print("Decoded masks match!")

assert_tensor_values_equal(
    target_encoder_output_olmo["tokens_and_masks"].sentinel2_l2a,
    target_output["sentinel2_l2a"],
    "tokens",
)
print("Target encoder tokens match!")

assert_tensor_values_equal(
    target_encoder_output_olmo["tokens_and_masks"].sentinel2_l2a_mask,
    target_output["sentinel2_l2a_mask"],
    "masks",
)
print("Target encoder masks match!")

assert_tensor_values_equal(
    target_encoder_output_olmo["project_aggregated"],
    target_output["pooled_tokens"],
    "pooled_tokens",
)
print("Target encoder pooled tokens match!")