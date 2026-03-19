import os
import argparse
import torch
import numpy as np
import random
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch import autocast
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from dataset.dataloader import load_and_prepare_dataset
from dataset.custom import ImageDataset

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_images(imgs, save_path, step):
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype("uint8")

    for idx, img in enumerate(imgs):
        Image.fromarray(img).save(
            os.path.join(save_path, f"image_{step}_{idx}.png")
        )

def sample(args):
    seed_everything(args.seed)
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    save_path = os.path.join(
        args.save_path,
        args.model,
        args.dataset,
        args.method,
        str(args.num_inference_steps),
    )
    os.makedirs(save_path, exist_ok=True)

    print(f"Saving to: {save_path}")
    print("Loading UNet...")
    unet = torch.load(args.model_path, map_location=device)
    unet.to(device).eval()

    print("Loading scheduler...")
    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
    else:
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

    if args.VAE is True:
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )
        vae.to(device)
        vae.requires_grad_(False)
    
    _, dataloader = load_and_prepare_dataset(
        dataset_name=args.dataset,
        batch_size=args.bs,
        img_size=(args.height, args.width),
        data_dir=None,
        rep_dir=args.rep_dir,
    )

    total_steps = len(dataloader)

    for step, batch in enumerate(dataloader):

        if args.VAE is True:
            latents = torch.randn(
                (args.bs, 4, args.height // 8, args.width // 8),
                device=device,
            )

        else:
            latents = torch.randn(
                (args.bs, 3, args.height, args.width),
                device=device,
            )

        scheduler.set_timesteps(args.num_inference_steps)
        encoder_hidden_states = batch.to(device, dtype=dtype).unsqueeze(1)

        with autocast(device.type):
            for t in tqdm(scheduler.timesteps, desc=f"Batch {step+1}/{total_steps}"):
                with torch.no_grad():
                    noise_pred = unet(
                        latents, t, encoder_hidden_states
                    )["sample"]

                latents = scheduler.step(
                    noise_pred, t, latents
                )["prev_sample"]

        if args.VAE is True:
            latents = latents / 0.18215
            with torch.no_grad():
                imgs = vae.decode(latents).sample
        else:
            imgs = latents

        save_images(imgs, save_path, step)
        del latents, imgs, noise_pred, batch
        torch.cuda.empty_cache()

        print(f"Saved batch {step+1}/{total_steps}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Provide the path to the model checkpoint")
    parser.add_argument("--rep_dir", type=str, required=True, help="Provide the path to the representations folder")
    parser.add_argument("--VAE", type=str, default=False)
    parser.add_argument("--dataset", type=str, default="cifar-10")
    parser.add_argument("--save_path", type=str, default="./samples")
    parser.add_argument("--scheduler", type=str, default="ddpm")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample(args)
