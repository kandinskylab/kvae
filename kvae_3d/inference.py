import glob
from argparse import ArgumentParser
from collections import defaultdict
import os
import datetime
from tqdm import tqdm

from omegaconf import OmegaConf
import torch
torch._dynamo.config.cache_size_limit = 10000

from models.cached_model import CachedCausalVAE
from models.efficient_vae import KandinskyVAE
from utils.inference_utils import compute_psnr_range
from utils.dataset_utils import make_png_tensor, make_video_tensor
from utils.utils import save

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


@torch.no_grad()
def infer(video_path, model, out_dir, frames=17, input_norm='m11', resize_input=-1, divider=8):
    if video_path.endswith('.mp4'):
        input_tensor, real_len = make_video_tensor(video_path, input_norm=input_norm, frames_cnt=frames, resize_input=resize_input, divider=divider)
    elif os.path.isdir(video_path):
        pngs_path = [os.path.join(video_path, filename) for filename in os.listdir(video_path) if filename.endswith('.png')]
        pngs_path.sort()
        input_tensor, real_len = make_png_tensor(pngs_path, input_norm=input_norm, frames_cnt=frames, resize_input=resize_input, divider=divider)
    else:
        raise ValueError
    
    input_tensor = input_tensor.unsqueeze(0)[:, :, :frames]
    *_, orig_h, orig_w = input_tensor.shape

    # Make input resolution dividable by divider
    input_tensor = input_tensor[:, :, :, :orig_h // divider * divider, :orig_w // divider * divider]

    input_tensor = input_tensor.to(DEVICE).to(DTYPE)
    
    with torch.no_grad():
        start_time = datetime.datetime.now()
        latent_and_params = model.encode(input_tensor)
        torch.cuda.synchronize()
        time_spent = datetime.datetime.now() - start_time
        print("Encode time: %f sec" % time_spent.total_seconds())

        if not isinstance(latent_and_params, tuple):
            latent_and_params = (latent_and_params,)
        if len(latent_and_params) > 2:
            latent, split_list, stat_params = latent_and_params
            latent_and_params = (latent, split_list)

        start_time = datetime.datetime.now()
        recs = model.decode(*latent_and_params)
        torch.cuda.synchronize()
        time_spent = datetime.datetime.now() - start_time
        print("Decode time: %f sec" % time_spent.total_seconds())

    input_tensor = input_tensor[:, :, :real_len]
    recs = recs[:, :, :real_len]

    # Crop to original shape (if needed)
    # *_, rec_h, rec_w = recs.shape
    # dh = rec_h - orig_h
    # dw = rec_w - orig_w
    # if dh > 0:
    #     recs = recs[..., dh//2:-dh//2, :]
    # if dw > 0:
    #     recs = recs[..., dw//2:-dw//2]
    
    torch.testing.assert_close(recs.shape, input_tensor.shape)
    
    recs = recs.float()
    psnr_range = compute_psnr_range(recs, input_tensor.float(), input_norm=input_norm)
    avg_psnr = psnr_range.mean()
    out_path = os.path.join(out_dir, os.path.basename(video_path).split('.')[0] + f'_psnr_{avg_psnr:.2f}')
    os.makedirs(out_path, exist_ok=True)
    save(out_path, torch.cat((input_tensor.float(), recs), dim=-1), input_norm=input_norm)
    return avg_psnr, psnr_range
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--frames', type=int, default=257)
    parser.add_argument('--optim', action="store_true")
    args = parser.parse_args()

    MY_DIR = '.'

    if torch.cuda.is_available():
        DTYPE = torch.bfloat16
        DEVICE = torch.device(f'cuda:{args.device_id}')
    else:
        DTYPE = torch.float
        DEVICE = torch.device("cpu")


    vae_class = KandinskyVAE if args.optim else CachedCausalVAE
    vae = vae_class.from_pretrained("kandinskylab/KVAE-3D-1.0")

    vae = vae.eval().to(DEVICE).to(DTYPE)

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    REC_DIR = os.path.join(MY_DIR, f'output/vae_recs/{vae.config["common"]["experiment_name"]}/{date}')

    testset = ('test1', 'test2')
    
    all_frames_psnr = list()

    for subdir in testset:
        log_dict = defaultdict(list)
        try:
            DATA_ROOT = os.path.join(MY_DIR, 'assets/%s' % subdir)
    
            psnr_sum = 0
            psnr_tested = 0

            data = sorted(os.listdir(DATA_ROOT))

            rec_dir = os.path.join(REC_DIR + '_fr%d%s' % (
                args.frames, 
                "_opt" if args.optim else '',
            ), subdir)
            exists = glob.glob(rec_dir + '_*')
            if exists:
                print("Already exists, skip ", exists)
                continue
            for name in tqdm(data):
                torch.cuda.empty_cache()
                v_path = os.path.join(DATA_ROOT, name)

                video_psnr, frames_psnr = infer(v_path, vae, rec_dir, 
                               frames=args.frames if args.frames != 999 else None, 
                               divider=vae.config["model"]["encoder_params"]["temporal_compress_times"] * 2)

                if not frames_psnr.shape:
                    frames_psnr = frames_psnr.unsqueeze(0)
                all_frames_psnr.append(frames_psnr)
                
                psnr_sum += video_psnr
                psnr_tested += 1
            avrg_psnr = psnr_sum / psnr_tested if psnr_tested else 0
            print("Average PSNR for %s: %0.2f" %(subdir, avrg_psnr))
            
            os.rename(rec_dir, rec_dir + "_psnr_%0.2f" % avrg_psnr)

        except Exception as e:
            print("Failed to process %s: %s" % (subdir, e))
            raise e
        
    print("PSNR total average: %f" % torch.cat(all_frames_psnr).mean())
