import torch
from diffusers import FluxPipeline
from dfloat11 import DFloat11Model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--prompt', type=str, default='A futuristic cityscape at sunset, with flying cars, neon lights, and reflective water canals')
parser.add_argument('--save_path', type=str, default='./image.png')
args = parser.parse_args()

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

DFloat11Model.from_pretrained('DFloat11/FLUX.1-dev-DF11', device='cpu', bfloat16_model=pipe.transformer)

prompt = args.prompt
image = pipe(
    prompt,
    width=1920,
    height=1440,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator(device="cuda").manual_seed(0)
).images[0]

image.save(args.save_path)
