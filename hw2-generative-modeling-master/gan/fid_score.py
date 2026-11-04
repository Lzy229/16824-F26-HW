import torch

from utils import get_fid
from networks import Generator

path_model = "./data_wgan_gp/generator.pt"

gen = Generator()

gen = torch.jit.load(
    path_model,
    map_location="cuda",
)


print("load successfully")
fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
print(f"Final FID (Full 50K): {fid}")