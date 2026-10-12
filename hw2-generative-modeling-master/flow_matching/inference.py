
import argparse
import os
import torch
import numpy as np
from torchvision.utils import save_image
from cleanfid import fid as cleanfid

from model import FlowModel
from unet import Unet  # reuse your existing UNet (it should return velocity field)

@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    ##################################################################
    # TODO 4.2: Write a function that samples images from the
    # flow matching model given z
    # Hint: Refer to diffusion/inference.py
    # Note: The output must be in the range [0, 255]!
    ##################################################################
    gen_fn = None

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    score = cleanfid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="train",
    )
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Matching Inference")
    parser.add_argument('--ckpt', required=True, type=str, help="Pretrained checkpoint")
    parser.add_argument('--num-images', default=100, type=int, help="Number of images per iteration")
    parser.add_argument('--image-size', default=32, type=int, help="Image size to generate")
    parser.add_argument('--solver', choices=['euler','rk4'], default='euler')
    parser.add_argument('--steps', type=int, default=50, help="Number of ODE steps to integrate")
    parser.add_argument('--compute-fid', action="store_true")
    args = parser.parse_args()

    prefix = f"flow_{args.solver}/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Build model (channels=3 for CIFAR-10). Reuse Unet architecture.
    model = Unet(
        dim=64,
        dim_mults=(1,2,4,8)
    ).cuda()

    flow = FlowModel(
        model,
        timesteps=1000,
        sampling_timesteps=args.steps,
        ode_solver=args.solver,
    ).cuda()

    # Load checkpoint (assumes you saved model_state_dict in training)
    ckpt = torch.load(args.ckpt, map_location='cuda')
    model.load_state_dict(ckpt["model_state_dict"])

    # Run sampling
    img_shape = (args.num_images, flow.channels, args.image_size, args.image_size)
    with torch.no_grad():
        model.eval()
        samples = flow.sample(img_shape, solver=args.solver, steps=args.steps)
        save_image(samples.data.float(), os.path.join(prefix, f"samples_{args.solver}.png"), nrow=10)

        if args.compute_fid:
            score = get_fid(flow, "cifar10", 32, 32*32*3, batch_size=256, num_gen=10_000)
            print("FID:", score)
