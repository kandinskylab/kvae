import os
import re
import random
import torchvision.transforms.functional as TF
import av
import torch
from PIL import Image
from glob import glob


import numpy as np


def get_urls(data_file):
    if os.path.isdir(data_file):
        urls = glob("%s/**/*.*" % data_file)
    else:
        with open(data_file) as f:
            urls = f.read().split('\n')
    return urls


def prepare_buffer(conf, global_rank):
    conf.nfs_buffer_path = os.path.join(conf.nfs_buffer_path, str(global_rank))
    os.makedirs(conf.nfs_buffer_path, exist_ok=True)


compiled_regexs = [
    (re.compile(r'the image depicts '), ''),
    (re.compile(r'the image features '), ''),
    (re.compile(r'the image is '), ''),
    (re.compile(r'the image captures '), ''),
    (re.compile(r'the image shows '), ''),
    (re.compile(r'the video is '), ''),
    (re.compile(r'the video features '), ''),
    (re.compile('<text>'), "'"),
    (re.compile('</text>'), "'"),
    (re.compile("стоковые фото и изображения"), ''),
    (re.compile("стоковое фото"), ''),
    (re.compile("cтоковое фото"), ''),
    (re.compile(r"[—-]?\s?stock illustrations?"), ''),
    (re.compile(r"[—-]?\s?stock images?"), ''),
    (re.compile(r' [-—] [^-]* стоковые фото и изображения'), ''),
    (re.compile(r' [-—] [^-]* stock illustrations?'), ''),
    (re.compile(r' [-—] [^-]* ст$'), ''),
    (re.compile(r'\s?[-—]?\s?[Ss]tock [Ii]llustrations?'), ''),
    (re.compile(r'\s?[-—]?\s?[CcСс]токовые [Фф]ото и [Ии]зображения'), ''),
    (re.compile(r'\s?[-—]?\s?[CcСс]токов[ыо]е [Фф]ото ?'), ''),
    (re.compile(r'\s?[-—]?\s?[Ss]tock [Ii]mages?'), ''),
    (re.compile(r'\s?[-—]?\s?[Ff]ree [Ss]tock [Pp]hotograph[yi]'), ''),
    (re.compile(r'\s?[-—]?\s?[Ss]tock [Pp]hotograph[yi]'), ''),
    (re.compile(r'\s?[-—]?\s?[Ss]tock [Pp]hotos?'), ''),
    (re.compile(r'\s?[-—]?\s?[Ss]tock [Vv]ideos?'), ''),
    (re.compile(r'\s?[-—]?\s?[Ss]tock [Mm]ozgókép?'), ''),
    (re.compile(r'shutterstock'), ''),
    (re.compile(r' ?[Кк]упить за \d+ руб\.?'), ''),
    (re.compile(r'[Ии]нтернет-магазин [Яя]рмарка [Мм]астеров\.?'), ''),
    (re.compile(r' ?[Яя]рмарка [Мм]астеров\.?'), ''),
    (re.compile('<person>'), '<unk>'),
    (re.compile(r'--FILE--'), ''),
    (re.compile(r'\.jpg'), ''),
    (re.compile(r'\.png'), ''),
    (re.compile(r'\.jpeg'), ''),
    (re.compile(r'\.svg'), ''),
    (re.compile(r'\.lux'), ''),
    (re.compile(r'\.pdf'), ''),
    (re.compile(r'\.tar'), ''),
    (re.compile(r'\.zip'), ''),
    (re.compile(r'\.txt'), ''),
    (re.compile(r'[Ll]oad [Ii]mage into [Gg]allery [Vv]iewer,? ?'), ''),
    (re.compile(r' - Property Image \d+'), ''),
    (re.compile(r' - SF33'), ''),
    (re.compile(r'【NEW】'), ''),
    (re.compile(r'\.\.\.'), '.'),
    (re.compile(r' \([Ii]mage \d+ of \d+\)'), ''),
    (re.compile(r' \([Vv]iew \d+ of \d+\)'), ''),
    (re.compile(r' \(?[Ii]mage number \d+\)?'), ''),
    (re.compile(r' ?\(?[Фф]ото [N,№]?\d+\)?'), ''),
    (re.compile(r' ?\(?\d+ ?[Фф]ото\)?\.?'), ''),
    (re.compile(r' ?\(?[Чч]асть \d+\)?\.?'), ''),
    (re.compile(r' ?[Ii]mg_\d+'), ''),
    (re.compile(r'[Pp]icture \d+ of \d+'), ''),
    (re.compile(r'[Ii]mage \d+ ?'), ''),
    (re.compile(r'[Pp]hoto \d+'), 'photo'),
    (re.compile(r'[Ss]creenshots? \d+'), 'screenshot'),
    (re.compile(r'( -)? pan left by <@\d+> \(.*\)$'), ''),
    (re.compile(r'( -)? pan right by <@\d+> \(.*\)$'), ''),
    (re.compile(r'( -)? remix by <@\d+> \(.*\)$'), ''),
    (re.compile(r'( -)? zoom in by <@\d+> \(.*\)$'), ''),
    (re.compile(r'( -)? zoom out by <@\d+> \(.*\)$'), ''),
    (re.compile(r'( -)? upscaled by <@\d+> \(.*\)$'), ''),
    (re.compile(r'( -)? (variations( \(.*\))? by )?<@\d+> \(.*\)$'), ''),
    (re.compile(r',? ?-? ?https?://.*$'), ''),
    (re.compile(r',? ?-? ?https?://.* '), ''),
    (re.compile(r' ?™'), ''),
    (re.compile(r' ?®'), ''),
    (re.compile(r'�'), ''),
    (re.compile(r' ?©'), ''),
    (re.compile(r' ?SC \d+'), ''),
    (re.compile(r' ?\(MLS #\w+\)'), ''),
    (re.compile(r' ?\w+\.com'), ''),
    (re.compile(r' ?\w+\.net'), ''),
    (re.compile(r' ?\w+\.org'), ''),
    (re.compile(r' ?\w+\.ru'), ''),
    (re.compile(r'&#\d+;'), ''),
    (re.compile(r' : \d+'), ''),
    (re.compile(r' ?\(\d+$'), ''),
    (re.compile(r'^[^\[]+\] ?'), ''),
    (re.compile(r' ?#\d+'), ''),
    (re.compile(r' \d+_\d+'), ''),
    (re.compile(r' [a-zA-Z]+\d+[a-zA-Z]+[\S]*'), ''),
    (re.compile(r' ?\(\d*\)'), ''),
    (re.compile(r' ?\([^\)]+$'), '')
]


def clean_with_regex(caption):
    lower_caption = str(caption).lower().strip() 
    for re_compiled, replacement in compiled_regexs: 
        iterator = reversed(list(re_compiled.finditer(lower_caption))) 
        for match in iterator: 
            pos = list(match.span()) 
            caption = caption[:pos[0]] + replacement + caption[pos[1]:]
            lower_caption = str(caption).lower().strip()
            
    if caption.count('-') > 2:
        split_captions = []
        for split_caption in caption.split():
            if split_caption.count('-') > 2:
                split_caption = re.sub(r'-', ' ', split_caption)
            split_captions.append(split_caption)
        caption = ' '.join(split_captions)
        
    caption = caption.strip('—-:/+=|@#&*')
        
    return caption.strip()


def preprocess_caption(data):
    caption_names = [
        'caption liuhaotian/llava-v1.5-13b prompt pixart', 'caption liuhaotian/llava-v1.5-13b prompt short',
        'caption liuhaotian/llava-v1.5-13b prompt detailed-long', 'caption lita-vicuna-v1-3-13b-finetune prompt detailed_video',
        'generated_captions', 'eng_caption_fixed', 'eng_caption', 'caption',
    ]
    for caption_name in caption_names:
        if caption_name in data and isinstance(data[caption_name], str):
            data['caption_clean'] = clean_with_regex(data[caption_name])
            return data
    data['caption_clean'] = ''
    return data



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