import torch


def compute_psnr_range(src, dist, input_type="image", clip: bool = True):
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
