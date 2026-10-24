import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    """
    Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): ReLU()
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    """
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        ##################################################################
        # TODO 2.1: Set up the network layers. First create the self.convs.
        # Then create self.fc with output dimension == self.latent_dim
        ##################################################################
        # the image shape after convs
        self.after_convs_shape = (256, int(self.input_shape[1] // 8), int(self.input_shape[2] // 8))
        self.after_convs_dim = 256 * self.after_convs_shape[1] * self.after_convs_shape[2]

        self.convs = nn.Sequential()

        self.convs.append(nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.convs.append(nn.ReLU())
        self.convs.append(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        self.convs.append(nn.ReLU())
        self.convs.append(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        self.convs.append(nn.ReLU())
        self.convs.append(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

        self.fc = nn.Linear(self.after_convs_dim, self.latent_dim, bias=True)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, output should be
        # of dimension == self.latent_dim
        ##################################################################
        convs_output = self.convs(x)

        B, C, H, W = convs_output.shape

        convs_output = convs_output.reshape(B, C*H*W)

        output = self.fc(convs_output)

        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        ##################################################################
        # TODO 2.4: Fill in self.fc, such that output dimension is
        # 2*self.latent_dim
        ##################################################################
        self.fc = nn.Linear(self.after_convs_dim, 2*self.latent_dim, bias=True)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.4: Forward pass through the network, should return a
        # tuple of 2 tensors, mu and log_std
        ##################################################################
        convs_output = self.convs(x)
        
        B, C, H, W = convs_output.shape

        convs_output = convs_output.reshape(B, C*H*W)
        
        latents = self.fc(convs_output)

        # print(latents.shape)
        
        mu = latents[:, :self.latent_dim]
        log_std = latents[:, self.latent_dim: ] 
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return mu, log_std


class Decoder(nn.Module):
    """
    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    """
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        ##################################################################
        # TODO 2.1: Set up the network layers. First, compute
        # self.base_size, then create the self.fc and self.deconvs.
        ##################################################################
        self.H = self.output_shape[1]
        self.W = self.output_shape[2]

        self.base_size = [int(self.H // 8), int(self.W // 8)]
        self.deconvs = nn.Sequential()

        self.deconvs.append(nn.ReLU())
        self.deconvs.append(nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
        self.deconvs.append(nn.ReLU())
        self.deconvs.append(nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
        self.deconvs.append(nn.ReLU())
        self.deconvs.append(nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
        self.deconvs.append(nn.ReLU())
        self.deconvs.append(nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))


        self.fc = nn.Linear(latent_dim, 256 * self.base_size[0] * self.base_size[1], bias=True)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, z):
        #TODO 2.1: forward pass through the network, 
        ##################################################################
        # TODO 2.1: Forward pass through the network, first through
        # self.fc, then self.deconvs.
        ##################################################################
        fc_output = self.fc(z)

        B = fc_output.shape[0]

        fc_output = torch.reshape(fc_output, (B, 256, self.base_size[0], self.base_size[1]))

        deconvs_output = self.deconvs(fc_output)

        return deconvs_output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    # NOTE: You don't need to implement a forward function for AEModel.
    # For implementing the loss functions in train.py, call model.encoder
    # and model.decoder directly.
