
import torch
from torchmetrics.image import PeakSignalNoiseRatio

psnr = None

def compute_psnr_range(src, dist, input_norm='01', uint8=True):
    global psnr
    if input_norm == '01':
        src.clip_(0, 1)
        src = (src * 255)
        if uint8:
            src = src.to(torch.uint8)
        dist.clip_(0, 1)
        dist = (dist * 255)
        if uint8:
            dist = dist.to(torch.uint8)
    elif input_norm == 'm11':
        src.clip_(-1, 127/128)
        src = ((src + 1) * 128)
        if uint8:
            src = src.to(torch.uint8)
        dist.clip_(-1, 127/128)
        dist = ((dist + 1) * 128)
        if uint8:
            dist = dist.to(torch.uint8)

    if psnr is None:
        psnr = PeakSignalNoiseRatio(data_range=(0, 255), reduction="none", dim=(0, 1, 3, 4)).cuda()

    return psnr(src, dist)




