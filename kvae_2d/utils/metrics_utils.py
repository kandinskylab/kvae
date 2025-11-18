import torch
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def compute_psnr_range(src, dist, input_type="video", clip: bool = True):
    src = (src.float() + 1) / 2
    dist = (dist.float() + 1) / 2
    if clip:
        src = src.clamp(0, 1)
        dist = dist.clamp(0, 1)
    mse = torch.nn.functional.mse_loss(src, dist, reduction="none")
    if input_type == "video":
        per_frame_mse = mse.mean(dim=(0, 1, 3, 4))
        per_frame_mse = per_frame_mse.view(-1).cpu().numpy()
        per_frame_psnr = -10 * torch.log10(per_frame_mse)
        return per_frame_psnr
    elif input_type == "image":
        return -10 * torch.log10(mse.mean())
    else:
        raise ValueError("input_type must be either 'video' or 'image'")


def lpips_score(recs, orig, lpips: LearnedPerceptualImagePatchSimilarity, scale_zero_one=False, split_size=32):
    if scale_zero_one:
        recs = recs.clip(0, 1)
        orig = orig.clip(0, 1)  # just because of eq transform with bicubic
    else:
        recs = recs.clip(-1, 1)
        orig = orig.clip(-1, 1)
    if orig.ndim == 5:
        orig = rearrange(orig, "b c t h w -> (b t) c h w")
        recs = rearrange(recs, "b c t h w -> (b t) c h w")
        orig = orig.split(split_size, dim=0)
        recs = recs.split(split_size, dim=0)
        results = []
        for a, b in zip(orig, recs):
            results.append(lpips(a, b))
        return torch.cat(results, dim=0).mean()
    elif orig.ndim == 4:
        return lpips(orig, recs)
    else:
        raise ValueError("Unexpected input shape")


def add_fid_images(fid: FrechetInceptionDistance, images, real: bool, scale_zero_one=False):
    images = images.float()
    if scale_zero_one:
        images = images.clip(0, 1)
    else:
        images = (images.clip(-1, 1) + 1) / 2
    if images.ndim == 5:
        images = rearrange(images, "b c t h w -> (b t) c h w")
    fid.update(images, real)
