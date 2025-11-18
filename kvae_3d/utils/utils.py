import functools
import multiprocessing as mp
import os

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torchvision


def load_conf(config_path, test=False):
    conf = OmegaConf.load(config_path)    

    return conf


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def save_one_frame(pair, out_dir):
    idx, np_img = pair
    out_path = os.path.join(out_dir, f'{idx:03}.png')
    Image.fromarray(np_img).save(out_path)


def save(out_path, recs, input_norm='m11', format='png'):
    recs = recs.squeeze(0).transpose(0, 1)
    recs = recs.float() 
    if input_norm == '01':
        recs.clip_(0, 1)
        recs = (recs.permute(0, 2, 3, 1).numpy(force=True) * 255).astype(np.uint8)
    elif input_norm == 'm11':
        recs.clip_(-1, 127/128)
        recs = ((recs.permute(0, 2, 3, 1).numpy(force=True) + 1) * 128).astype(np.uint8)
    else:
        raise NotImplementedError("Norm type %s is not supported" % input_norm)
    if format == 'png':
        save_frames = functools.partial(save_one_frame, out_dir=out_path)
        with mp.Pool(processes=4) as pool:
            pool.map(save_frames, enumerate(recs, start=1))
    elif format == 'mp4':
        torchvision.io.write_video(out_path, recs, fps=10, video_codec='h264')

