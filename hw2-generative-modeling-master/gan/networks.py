import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        n_filters=128,
        kernel_size=3,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.upscale_factor = upscale_factor
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel-wise upscale_factor^2 times
        # 2. Use torch.nn.PixelShuffle to form an output of dimension
        # (batch, channel, height*upscale_factor, width*upscale_factor)
        # 3. Apply convolution and return output
        ##################################################################
        # r is upscale_factor
        # assume the x size is (B, C, H, W)
        upsample_output = x.repeat_interleave(int(self.upscale_factor ** 2), dim=1)
        # the repeat result should be (B, C*r^2, H, W)

        shuffle_output = self.pixelshuffle(upsample_output)
        # the pixel_shuffle result would be (B, C, H*r, W*r)

        output = self.conv(shuffle_output)
        # pass through the convolution layer
        
        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, n_filters=128, kernel_size=3, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.downscale_ratio = downscale_ratio
        self.pixelunshuffle = nn.PixelUnshuffle(self.downscale_ratio)

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension
        # (batch, channel*downscale_factor^2, height, width)
        # hint: using view/reshape instead of an explicit split
        # 2. Then split channel-wise and reshape into
        # (downscale_factor^2, batch, channel, height, width) images
        # 3. Take the average across dimension 0, apply convolution,
        # and return the output
        ##################################################################
        # the input size should be (B, C, H*r, W*r)

        # the output size after unshuffle should be (B, C*r^2, H, W)
        unshuffle_output = self.pixelunshuffle(x)

        # get the size of B, C, H, W
        B, C, H, W = unshuffle_output.shape
        C = int(C // (self.downscale_ratio ** 2))

        # reshape the output to (d^2, B, C, H, W)
        reshape_output = torch.reshape(unshuffle_output, (int(self.downscale_ratio**2), B, C, H, W))

        # take average
        average_output = torch.mean(reshape_output, dim=0)

        # pass the convolution layer
        output = self.conv(average_output)
        
        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential()

        # (0): BatchNorm2d, eps, momentum, affine, track_running_stats are the same with default
        self.layers.append(nn.BatchNorm2d(input_channels))
        # (1): Relu:
        self.layers.append(nn.ReLU())
        # (2): Conv2d, the output_channels is n_filters
        self.layers.append(nn.Conv2d(input_channels, n_filters, kernel_size, stride=(1, 1), padding=(1, 1), bias=False))
        # (3): BatchNorm2d
        self.layers.append(nn.BatchNorm2d(n_filters))
        # (4): Relu:
        self.layers.append(nn.ReLU())
        # (5): UpsampleConv2D
        self.layers.append(UpSampleConv2D(n_filters, n_filters, kernel_size, padding=1))

        # upsample_residual
        self.upsample_residual = UpSampleConv2D(input_channels, n_filters, kernel_size=1)

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to upsample the residual before adding it
        # to the layer output.
        ##################################################################
        # pass through layers
        layers_output = self.layers(x)

        # pass through upsample residual
        upsample_output = self.upsample_residual(x)

        # add it up
        output = layers_output + upsample_output

        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential()

        # (1): Relu
        self.layers.append(nn.ReLU())
        # (2): Conv2d
        self.layers.append(nn.Conv2d(input_channels, n_filters, kernel_size, padding=1))
        # (3): Relu
        self.layers.append(nn.ReLU())
        # (4): DownSample
        self.layers.append(DownSampleConv2D(n_filters, n_filters, kernel_size, padding=1))

        # downsample_Residual
        self.downsample_residual = DownSampleConv2D(input_channels, n_filters, kernel_size=1)

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward through the layers and implement a residual
        # connection. Make sure to downsample the residual before adding
        # it to the layer output.
        ##################################################################
        # pass through layers
        layers_output = self.layers(x)

        # pass through upsample residual
        downsample_output = self.downsample_residual(x)

        # add it up
        output = layers_output + downsample_output

        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        ##################################################################
        # TODO 1.1: Setup the network layers
        ##################################################################
        self.layers = nn.Sequential()

        # (0): ReLU
        self.layers.append(nn.ReLU())
        # (1): Conv2d
        self.layers.append(nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=1))
        # (2): Relu
        self.layers.append(nn.ReLU())
        # (3): Conv2d
        self.layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=1))

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the conv layers. Don't forget the residual
        # connection!
        ##################################################################
        layers_output = self.layers(x)

        # residual connection
        output = layers_output + x

        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class Generator(torch.jit.ScriptModule):
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.dense = nn.Linear(in_features=128, out_features=2048, bias=True)
        
        # setup layers
        self.layers = nn.Sequential()
        # (0, 1, 2): ResBlockUp for three layers
        self.layers.append(ResBlockUp(input_channels=128))
        self.layers.append(ResBlockUp(input_channels=128))
        self.layers.append(ResBlockUp(input_channels=128))
        # (3): BatchNorm
        self.layers.append(nn.BatchNorm2d(num_features=128))
        # (4): Relu
        self.layers.append(nn.ReLU())
        # (5): Conv2d
        self.layers.append(nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1))
        # (6): Tanh
        self.layers.append(nn.Tanh())

        # starting_image_size
        self.starting_image_size = starting_image_size

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
        ##################################################################
        # TODO 1.1: Forward the generator assuming a set of samples z has
        # been passed in. Don't forget to re-shape the output of the dense
        # layer into an image with the appropriate size!
        ##################################################################
        # batch_size, Channel, Height, Width
        B = z.shape[0]

        dense_output = self.dense(z)

        # if torch.isnan(dense_output).any():
        #     print("gen is nan")

        # re-shape? 
        # after the dense layer, the channel from 128 to 2048, but the input
        # channel of layers is 128, so it need reshaoe?
        reshape_output = dense_output.reshape(B, 128, self.starting_image_size, self.starting_image_size)

        output = self.layers(reshape_output)

        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        ##################################################################
        # TODO 1.1: Generate n_samples latents and forward through the
        # network.
        ##################################################################
        # make sure the input channel is 128, so the batchsize is 1024/128 = 8
        # import pdb; pdb.set_trace()
        
        latents = torch.randn((n_samples, 128), dtype=torch.float16).cuda()

        # if torch.isnan(latents).any():
        #     print("latent is nan")

        output = self.forward_given_samples(latents)

        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        ##################################################################
        # TODO 1.1: Set up the network layers. You should use the modules
        # you have implemented previously above.
        ##################################################################
        self.dense = nn.Linear(in_features=128, out_features=1, bias=True)
        # the dense result maybe < 0 need a sigmoid to keep the output within (0, 1)
        self.sigmoid = nn.Sigmoid()
        
        # initial layers
        self.layers = nn.Sequential()
        # (0): ResBlockDown
        self.layers.append(ResBlockDown(input_channels=3))
        # (1): ResBlockDown
        self.layers.append(ResBlockDown(input_channels=128))
        # (2): ResBlock
        self.layers.append(ResBlock(input_channels=128))
        # (3): ResBlock
        self.layers.append(ResBlock(input_channels=128))
        # (4): Relu
        self.layers.append(nn.ReLU())

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # TODO 1.1: Forward the discriminator assuming a batch of images
        # have been passed in. Make sure to sum across the image
        # dimensions after passing x through self.layers.
        ##################################################################

        layers_output = self.layers(x)

        # if torch.isnan(layers_output).any():
        #     print("layers in discriminater nan")

        # print(layers_output.shape)
        B, C, H, W = layers_output.shape
        layers_output = layers_output.reshape((B * H * W, C))
        # layers_output = layers_output.sum(dim=(2, 3))

        dense_output = self.dense(layers_output)

        # if torch.isnan(dense_output).any():
        #     print("dense in discriminater nan")

        output = self.sigmoid(dense_output)

        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
