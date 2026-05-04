from WIS_DSP_lib.dataloader import Dataset, DataLoader
from HW03 import Encoder
from WIS_DSP_lib.constants import *
from WIS_DSP_lib.olmo_helper import get_dataloader, get_encoder
from WIS_DSP_lib.test_helpers import assert_tensor_values_equal

seed = 3622

#%% Run the olmoearth_pretrain code 
set_reproducible_seeds(seed)

# load batch
data_loader, dataset = get_dataloader()

data_iterator = iter(data_loader)
batch = next(data_iterator)
encoder = get_encoder()
# encode
output_olmo = encoder(batch[1], patch_size=8)

#%% Run our refactored code
set_reproducible_seeds(seed)
dataset = Dataset(h5py_dir=DATA_DIR)
data_loader = DataLoader(
        dataset=dataset,
        batch_size=GLOBAL_BATCH_SIZE,
    )
batch = next(iter(data_loader))
encoder = Encoder(MODALITY, embedding_size=128, max_patch_size=8, num_heads=8, depth=4, mlp_ratio=4.0)
output = encoder(batch[0])

assert_tensor_values_equal(
        output_olmo["tokens_and_masks"].sentinel2_l2a,
        output[MODALITY],
        "tokens",
    )

assert_tensor_values_equal(
        output_olmo["tokens_and_masks"].sentinel2_l2a_mask,
        output["sentinel2_l2a_mask"],
        "masks",
    )

assert_tensor_values_equal(
        output_olmo["project_aggregated"],
        output["pooled_tokens"],
        "pooled_tokens",
    )