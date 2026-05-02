import os 
import torch 
import torch
from torch_geometric.data import Data

from src.cdvae_model import CDVAE, NUM_ELEMENTS, MAX_ATOMS, params_to_lattice
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "cdvae_checkpoint_epoch_110.pt")
checkpoint = torch.load(CKPT_PATH, map_location="cpu")
model = CDVAE()
#model.load_state_dict(checkpoint["model_state"])
#print(model["model_state"])
print(checkpoint["history"][90]["loss"])
data_path = "L:\\aj_cdvae\\aj051k\\dataset_files\\batch_0000.pt"
data = torch.load(data_path,weights_only=False)
#one batch has 100 datapoints and each for each we have attributees , we use data[0] to read it 
# y = formation energy 
print(data[0])