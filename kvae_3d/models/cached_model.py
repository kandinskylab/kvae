import math

import torch

from .cached_enc_dec import CachedEncoder3D, CachedDecoder3D


class CachedCausalVAE(torch.nn.Module):
    def __init__(self, encoder_conf, decoder_conf, ckpt_path=None):
        super().__init__()
        self.conf = {'enc' : encoder_conf,
                     'dec' : decoder_conf}
        self.encoder = CachedEncoder3D(**encoder_conf)
        self.decoder = CachedDecoder3D(**decoder_conf)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]

        # Fix checkpoint if starting from new style
        replace_keys = dict()
        for k in sd:
            if 'encoder.down' in k and 'downsample.temporal_conv.conv' in k:
                continue
            if k.startswith('loss'):
                replace_keys[k] = None

            if k.startswith('decoder'):
                if 'upsample' in k:
                    continue
                elif '.conv_b.conv' in k:
                    replace_keys[k] = k.replace('.conv_b.conv', '.conv_b')
                    continue
                elif '.conv_y.conv' in k:
                    replace_keys[k] = k.replace('.conv_y.conv', '.conv_y')
                    continue
                
            if k.endswith('sample.temporal_conv.conv.weight') or k.endswith('sample.temporal_conv.conv.bias'):
                replace_keys[k] = k.replace(".temporal_conv.conv.", '.temporal_conv.')
        for old_k, new_k in replace_keys.items():
            if new_k is not None:
                sd[new_k] = sd[old_k]
            del sd[old_k]

        self.load_state_dict(sd, strict=True)
        print(f'Restored weights from {path}')

    def make_empty_cache(self, block: str):
        def make_dict(name, p=None):
            if name == 'conv':
                return {'padding' : None}

            layer, module = name.split('_')
            if layer == 'norm':
                if module == 'enc':
                    return {'mean' : None,
                            'var' : None}
                else:
                    return {'norm' : make_dict('norm_enc'),
                            'add_conv' : make_dict('conv')}
            elif layer == 'resblock':
                return {'norm1' : make_dict(f'norm_{module}'),
                        'norm2' : make_dict(f'norm_{module}'),
                        'conv1' : make_dict('conv'),
                        'conv2' : make_dict('conv'),
                        'conv_shortcut' : make_dict('conv')}
            elif layer.isdigit():
                out_dict = {'down' : [make_dict('conv'), make_dict('conv')],
                            'up' : make_dict('conv')}
                for i in range(p):
                    out_dict[i] = make_dict(f'resblock_{module}')

                return out_dict

        cache = {'conv_in' : make_dict('conv'),
                 'mid_1' : make_dict(f'resblock_{block}'),
                 'mid_2' : make_dict(f'resblock_{block}'),
                 'norm_out' : make_dict(f'norm_{block}'),
                 'conv_out' : make_dict('conv')}
        for i in range(len(self.conf[block].get("ch_mult", [1, 2, 4, 8]))):
            cache[i] = make_dict(f'{i}_block', p=self.conf[block].num_res_blocks+1)
        return cache

    def encode(self, x, seg_len=16):
        cache = self.make_empty_cache('enc')

        ## get segments size
        split_list = [seg_len + 1]
        n_frames = x.size(2) - (seg_len + 1)
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len

        split_list[-1] += n_frames

        ## encode by segments
        latent = []
        params = []

        for chunk in torch.split(x, split_list, dim=2):
            l = self.encoder(chunk, cache)
            sample, _ = torch.chunk(l, 2, dim=1)
            latent.append(sample)
            params.append(l)

        latent = torch.cat(latent, dim=2)
        return latent, split_list, params

    def decode(self, z, split_list):
        cache = self.make_empty_cache('dec')

        ## get segments size
        split_list = [math.ceil(size / self.conf["enc"]["temporal_compress_times"]) for size in split_list]

        ## decode by segments
        recs = []
        for chunk in torch.split(z, split_list, dim=2):
            out = self.decoder(chunk, cache)
            recs.append(out)

        recs = torch.cat(recs, dim=2)
        return recs

    def forward(self, x, seg_len: int=16):
        latent, split_list = self.encode(x, seg_len)
        recs = self.decode(latent, split_list)
        return recs
