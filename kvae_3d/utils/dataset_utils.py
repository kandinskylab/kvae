import os
import re
import random
import torchvision.transforms.functional as TF
import av
import torch
from PIL import Image
from glob import glob


import numpy as np


RESOLUTIONS = {
    256: [(256, 256), (256, 384), (384, 256)],
    512: [(512, 512), (512, 768), (768, 512)],
    1024: [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)],
    0: [
    (256, 256), (256, 384), (384, 256),
    (512, 512), (512, 768), (768, 512),
    (1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)
    ]
}

def find_nearest_size(h, w, resolution=0):
    nearest_index = np.argmin(
        [
            *map(
                lambda x: abs((x[0] / x[1]) - (h / w)) + 1 / w / h, # Prefer larger resolutions
                RESOLUTIONS[resolution],
            )
        ]
    )
    return RESOLUTIONS[resolution][nearest_index]

def resize_img(img, resolution=0):
    w, h = img.size
    nw, nh = find_nearest_size(w, h, resolution)
    scale_factor = min(h / nh, w / nw)
    img = img.resize((int(w / scale_factor), int(h / scale_factor)), Image.Resampling.BICUBIC)

    w, h = img.size
    img = img.crop(((w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh))
    return img


def make_video_tensor(video_path, frames_cnt=None, input_norm='01', resize_input=-1, divider=8):
    input_tensor = []
    with av.open(video_path) as video_reader:
        video_length = video_reader.streams.video[0].frames

        for idx, frame in enumerate(video_reader.decode(video=0)):
            pil_img = frame.to_rgb().to_image()

            if resize_input >= 0:
                pil_img = resize_img(pil_img, resize_input)

            np_img = np.asarray(pil_img)
            if input_norm == '01':
                np_img = np_img / 255.
            elif input_norm == 'm11':
                np_img = np_img / 128 - 1
            else:
                raise NotImplementedError("Norm type %s is not supported" % input_norm)
            tensor_img = TF.to_tensor(np_img) # .mul_(2).add_(-1)
            input_tensor.append(tensor_img)
            if frames_cnt is not None and len(input_tensor) == frames_cnt:
                break

    # real len may be less than frames_cnt
    real_len = len(input_tensor)
    # Pad to required len
    if frames_cnt is not None:
        while len(input_tensor) < frames_cnt:
            input_tensor.append(input_tensor[-1])
    else:
        # Pad to x8
        while len(input_tensor) % divider != 1:
            input_tensor.append(input_tensor[-1])

    input_tensor = torch.stack(input_tensor, dim=1)
    return input_tensor, real_len


def make_png_tensor(png_list, frames_cnt=None, input_norm='01', resize_input=-1, divider=8):
    input_tensor = []
    for png in png_list:
        with Image.open(png).convert('RGB') as pil_img:

            if resize_input > 0:
                pil_img = resize_img(pil_img, resize_input)

            np_img = np.asarray(pil_img)
            if input_norm == '01':
                np_img = np_img / 255.
            elif input_norm == 'm11':
                np_img = np_img / 128 - 1
            else:
                raise NotImplementedError("Norm type %s is not supported" % input_norm)
            tensor_img = TF.to_tensor(np_img) # .mul_(2).add_(-1)
            input_tensor.append(tensor_img)
        if frames_cnt is not None and len(input_tensor) == frames_cnt:
            break    

    # real len may be less than frames_cnt
    real_len = len(input_tensor)

    # Pad to required len
    if frames_cnt is not None:
        while len(input_tensor) < frames_cnt:
            input_tensor.append(input_tensor[-1])
    else:
        # Pad to x8
        while len(input_tensor) % divider != 1:
            input_tensor.append(input_tensor[-1])

    input_tensor = torch.stack(input_tensor, dim=1)
    return input_tensor, real_len