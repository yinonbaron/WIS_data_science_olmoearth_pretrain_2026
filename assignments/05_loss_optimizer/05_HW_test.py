from olmo_core.utils import get_default_device
from WIS_DSP_lib.dataloader import Dataset, DataLoader
from HW05 import Model, TrainModule
from WIS_DSP_lib.constants import *
from WIS_DSP_lib.olmo_helper import get_dataloader, get_encoder, get_decoder, get_trainer
from WIS_DSP_lib.test_helpers import assert_tensor_values_equal, assert_values_equal
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from tqdm import tqdm

device = get_default_device()
seed = 3622

#%% Run the olmoearth_pretrain code 
set_reproducible_seeds(seed)

# load batch
data_loader, dataset = get_dataloader(device=device)
encoder = get_encoder()
decoder = get_decoder()
model = LatentMIM(encoder, decoder)
trainer = get_trainer(model, device, data_loader)
total_steps = 2
step = 0
olmo_losses = []
for batch in tqdm(iter(data_loader)):
    loss = trainer.train_module.train_batch(batch)
    
    trainer.train_module.optim_step()
    trainer.train_module.zero_grads()

    step += 1
    if step >= total_steps:
        print(step)
        break
    olmo_losses.append(loss.cpu().detach().numpy())

##% Run our refactored code
set_reproducible_seeds(seed)
dataset = Dataset(h5py_dir=DATA_DIR)
data_loader = DataLoader(
        dataset=dataset,
        batch_size=GLOBAL_BATCH_SIZE,
    )

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
model = Model(encoder_config, decoder_config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
train_module = TrainModule(model=model, dataloader=data_loader, optimizer=optimizer)

losses = []; total_steps = 2; step = 0;
for batch in tqdm(iter(train_module.dataloader)):
            train_module.optimizer.zero_grad()
            loss = train_module.train_batch(batch)
            train_module.optim_step()
            losses.append(loss.cpu().detach().numpy())
            step += 1
            if step >= total_steps:
                break

for index, (olmo_loss, standalone_loss) in enumerate(zip(olmo_losses, losses)):
        assert_values_equal(olmo_loss, standalone_loss, f"losses_{index}")
