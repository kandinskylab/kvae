import torch.nn as nn

from .layers import ResnetBlock2D, nonlinearity, PXSUpsample
from .norm_layer import DecoderSpacialNorm2D


class Decoder2D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        in_channels,
        z_channels,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        if zq_ch is None:
            zq_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = nn.Conv2d(
            in_channels=z_channels, out_channels=block_in, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=DecoderSpacialNorm2D,
        )

        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=DecoderSpacialNorm2D,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        normalization=DecoderSpacialNorm2D,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = PXSUpsample(in_channels=block_in)
            self.up.insert(0, up)

        self.norm_out = DecoderSpacialNorm2D(block_in, zq_ch, add_conv=add_conv)  # , gather=gather_norm)

        self.conv_out = nn.Conv2d(
            in_channels=block_in, out_channels=out_ch, kernel_size=3, padding=(1, 1), padding_mode="replicate"
        )

    def forward(self, z, clear_fake_cp_cache=True, use_cp=True):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        t = z.shape[2]
        # z to block_in

        zq = z
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.block_2(h, temb, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)

                # h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    def get_last_layer(self):
        return self.conv_out.weight
