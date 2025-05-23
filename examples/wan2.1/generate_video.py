from argparse import ArgumentParser
import time

import torch

from dfloat11 import DFloat11Model
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


parser = ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='Wan-AI/Wan2.1-T2V-14B-Diffusers')
parser.add_argument('--dfloat11_model_name_or_path', type=str, default='DFloat11/Wan2.1-T2V-14B-Diffusers-DF11')
parser.add_argument('--prompt', type=str, default='A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.')
parser.add_argument('--negative_prompt', type=str, default='Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards')
parser.add_argument('--resolution', type=str, choices=['480p', '720p'], default='480p')
parser.add_argument('--num_frames', type=int, default=33)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--guidance_scale', type=float, default=5.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_file_name', type=str, default='output.mp4')
args = parser.parse_args()

vae = AutoencoderKLWan.from_pretrained(args.model_name_or_path, subfolder="vae", torch_dtype=torch.float32)
scheduler = UniPCMultistepScheduler(
    prediction_type='flow_prediction',
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=3.0 if args.resolution == '480p' else 5.0
)

pipe = WanPipeline.from_pretrained(args.model_name_or_path, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.enable_model_cpu_offload()

DFloat11Model.from_pretrained(args.dfloat11_model_name_or_path, device='cpu', bfloat16_model=pipe.transformer)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start_time = time.time()
output = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    height=480 if args.resolution == '480p' else 720,
    width=832 if args.resolution == '480p' else 1280,
    num_frames=args.num_frames,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    generator=torch.Generator(device="cuda").manual_seed(args.seed),
).frames[0]
end_time = time.time()

peak_allocated = torch.cuda.max_memory_allocated() / 1e9
print(f'GPU Peak Memory Usage: {peak_allocated:.2f} GB')
print(f'Video Generation Time: {end_time - start_time:.2f} seconds')

export_to_video(output, args.output_file_name, fps=16)
