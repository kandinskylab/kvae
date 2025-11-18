import torch.nn as nn

from .autoencoder import AutoEncoder
from .regularizers import GaussianPrior
from .decoder import Decoder2D
from .encoder import Encoder2D


class KVAE2D(AutoEncoder):
    def __init__(
        self,
        in_channels=3,
        channels=128,
        num_enc_blocks=2,
        num_dec_blocks=2,
        z_channels=16,
        double_z=True,
        ch_mult=(1, 2, 4, 8),
        bottleneck: nn.Module = None,
    ):
        encoder = Encoder2D(
            in_channels=in_channels,
            ch=channels,
            ch_mult=ch_mult,
            num_res_blocks=num_enc_blocks,
            z_channels=z_channels,
            double_z=double_z,
        )
        decoder = Decoder2D(
            out_ch=in_channels,
            ch=channels,
            ch_mult=ch_mult,
            num_res_blocks=num_dec_blocks,
            in_channels=None,
            z_channels=z_channels,
        )
        posterior = bottleneck or GaussianPrior()
        super().__init__(encoder=encoder, decoder=decoder, bottleneck=posterior)
