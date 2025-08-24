from argparse import ArgumentParser

import torch
from diffusers import FluxPipeline

from dfloat11 import compress_model


parser = ArgumentParser("Compress FLUX.1 model using DFloat11")
parser.add_argument(
    '--model_name_or_path',
    type=str,
    default='black-forest-labs/FLUX.1-dev',
    choices=['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell', 'black-forest-labs/FLUX.1-Krea-dev'],
    help='The name or path of the FLUX.1 model to compress'
)
parser.add_argument(
    '--save_path',
    type=str,
    default='./FLUX.1-dev-DF11',
    help='The path to save the compressed model'
)
parser.add_argument(
    '--save_single_file',
    action='store_true',
    help='Save the compressed model as a single .safetensorsfile'
)
parser.add_argument(
    '--check_correctness',
    action='store_true',
    help='Check the correctness of the compressed weights during compression'
)
parser.add_argument(
    '--block_range',
    type=int,
    nargs=2,
    default=(0, 100),
    help='The range of transformer blocks to compress (for parallel compression over multiple CPU cores)'
)
args = parser.parse_args()

# Load the FLUX.1 model in bfloat16 precision
pipe = FluxPipeline.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
model = pipe.transformer

# Compress the model using DFloat11 compression
compress_model(
    model=model,
    pattern_dict={
        "transformer_blocks\.\d+": (
            "norm1.linear",
            "norm1_context.linear",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.add_q_proj",
            "attn.to_out.0",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ),
        "single_transformer_blocks\.\d+": (
            "norm.linear",
            "proj_mlp",
            "proj_out",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
        ),
    },
    save_path=args.save_path,
    save_single_file=args.save_single_file,
    check_correctness=args.check_correctness,
    block_range=args.block_range,
)
