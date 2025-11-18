import torch.nn as nn

from .layers import get_norm_layer_2d, nonlinearity, ResnetBlock2D, PXSDownsample


class Encoder2D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        in_channels,
        z_channels,
        double_z=True,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * self.num_resolutions
        else:
            self.num_res_blocks = num_res_blocks

        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.ch,
            kernel_size=3,
            padding=(1, 1),
        )

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_resolutions - 1:
                down.downsample = PXSDownsample(in_channels=block_in)  # mb: bad out channels
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
        )

        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
        )

        # end
        self.norm_out = get_norm_layer_2d(block_in)

        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=(1, 1),
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    def get_last_layer(self):
        return self.conv_out.weight
