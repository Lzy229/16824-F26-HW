import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    # print(discrim_real.shape, discrim_fake.shape)
    # import pdb; pdb.set_trace()
    
    B = discrim_real.shape[0]

    target_fake = torch.zeros_like(discrim_fake).cuda()
    target_real = torch.ones_like(discrim_real).cuda()

    fake_loss = F.binary_cross_entropy_with_logits(discrim_fake, target_fake)
    real_loss = F.binary_cross_entropy_with_logits(discrim_real, target_real)

    # log_disc_fake = torch.clamp(torch.log(1-discrim_fake), -100, 0)
    # log_disc_real = torch.clamp(torch.log(discrim_real), -100, 0)

    # loss = -torch.sum(log_disc_real / B) - torch.sum(log_disc_fake / B)

    # # loss = (fake_loss + real_loss) / 2
    # if torch.isnan(loss):
    #     import pdb; pdb.set_trace()
    # if torch.isinf(loss):
    #     import pdb; pdb.set_trace()

    loss = fake_loss + real_loss
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    B = discrim_fake.shape[0]

    target_fake = torch.ones_like(discrim_fake).cuda()

    loss = F.binary_cross_entropy_with_logits(discrim_fake, target_fake)
    
    # log_disc_fake = torch.clamp(torch.log(discrim_fake), -100, 0)

    # loss = -torch.sum(log_disc_fake / B)

    # if torch.isnan(loss):
    #     import pdb; pdb.set_trace()
    # if torch.isinf(loss):
    #     import pdb; pdb.set_trace()

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
