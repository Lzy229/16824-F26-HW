import torch

from utils import interpolate_latent_space
from networks import Generator

path_model = "./data_gan/generator.pt"
path_interpolate = "./data_gan/interpolations_30000.png"

gen = Generator()

gen = torch.jit.load(
    path_model,
    map_location="cuda",
)


print("load successfully")

interpolate_latent_space(gen, path_interpolate)
print("generate successfully")