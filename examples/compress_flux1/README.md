# Example: Compressing FLUX.1 Models Using DFloat11

This example demonstrates how to compress FLUX.1 models using the DFloat11 compression framework. DFloat11 achieves approximately 32% model size reduction while maintaining bit-for-bit identical outputs to the original model.

## Requirements

**System Requirements:**
- CUDA-compatible GPU (for correctness checking)
- Python 3.9+
- PyTorch installed

Install the required dependencies:
```bash
pip install -U diffusers dfloat11[cuda12]
```

## Usage

### Single-Core Compression

To compress the `FLUX.1-dev` model into a single safetensors file:

```bash
python compress_flux.py \
    --model_name_or_path black-forest-labs/FLUX.1-dev \
    --save_path ./FLUX.1-dev-DF11 \
    --save_single_file \
    --check_correctness
```

**Command Line Arguments**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name_or_path` | str | `black-forest-labs/FLUX.1-dev` | HuggingFace model name of the FLUX.1 model. Supports `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell`, and `black-forest-labs/FLUX.1-Krea-dev` |
| `--save_path` | str | `./FLUX.1-dev-DF11` | Directory path where the compressed model will be saved |
| `--save_single_file` | flag | False | Save as a single `.safetensors` file instead of multiple `.safetensors` shards |
| `--check_correctness` | flag | False | Verify the compressed weights are bit-for-bit identical to the original weights |
| `--block_range` | int, int | `(0, 100)` | Range of transformer blocks to compress (for parallel processing) |

### Multi-Core Parallel Compression

For faster compression using multiple CPU cores, run:

```bash
bash compress_flux_parallel.sh
```
> **Note:** Parallel compression requires substantial CPU memory and produces multiple safetensors shards instead of a single file.

The `compress_flux_parallel.sh` script automatically splits the work across CPU cores. By default:

* Each CPU core compresses **5 transformer blocks**.
* A total of **12 cores** are used to compress the **57 blocks** of FLUX.1.

To change how many blocks each core processes, edit the `BLOCKS_PER_TASK` variable in the script.

## Next Steps

After compression, you can run the compressed model with the following Python script (20GB VRAM required):

```python
import torch
from diffusers import FluxPipeline
from dfloat11 import DFloat11Model

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
DFloat11Model.from_pretrained('./FLUX.1-dev-DF11', device='cpu', bfloat16_model=pipe.transformer)
pipe.enable_model_cpu_offload()

prompt = "A futuristic cityscape at sunset, with flying cars, neon lights, and reflective water canals"
image = pipe(
    prompt,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator(device="cuda").manual_seed(0)
).images[0]

image.save("image.png")
```
---

## Understanding the Pattern Dictionary

DFloat11 decompresses weights at the **transformer-block level**, compressing and decompressing all weights in a block as a single unit. This approach maximizes decompression throughput and reduces runtime overhead.

The `pattern_dict` defines how modules are grouped for compression. It is a Python dictionary where:

* **Key** – a regex pattern matching transformer-block module names.
* **Value** – a list of linear submodules within each matched block that are compressed and decompressed together.

For example, the PyTorch FLUX.1-dev model uses the following structure and corresponding `pattern_dict`:

### FLUX.1 Model Structure

When you print out a PyTorch FLUX.1 model, you'll see this structure:

```python
FluxTransformer2DModel(
  (pos_embed): FluxPosEmbed()
  (time_text_embed): CombinedTimestepGuidanceTextProjEmbeddings(
    (time_proj): Timesteps()
    (timestep_embedder): TimestepEmbedding(
      (linear_1): Linear(in_features=256, out_features=3072, bias=True)
      (act): SiLU()
      (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
    )
    (guidance_embedder): TimestepEmbedding(
      (linear_1): Linear(in_features=256, out_features=3072, bias=True)
      (act): SiLU()
      (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
    )
    (text_embedder): PixArtAlphaTextProjection(
      (linear_1): Linear(in_features=768, out_features=3072, bias=True)
      (act_1): SiLU()
      (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
    )
  )
  (context_embedder): Linear(in_features=4096, out_features=3072, bias=True)
  (x_embedder): Linear(in_features=64, out_features=3072, bias=True)
  (transformer_blocks): ModuleList(
    (0-18): 19 x FluxTransformerBlock(
      (norm1): AdaLayerNormZero(
        (silu): SiLU()
        (linear): Linear(in_features=3072, out_features=18432, bias=True)
        (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      )
      (norm1_context): AdaLayerNormZero(
        (silu): SiLU()
        (linear): Linear(in_features=3072, out_features=18432, bias=True)
        (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      )
      (attn): FluxAttention(
        (norm_q): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (norm_k): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (to_q): Linear(in_features=3072, out_features=3072, bias=True)
        (to_k): Linear(in_features=3072, out_features=3072, bias=True)
        (to_v): Linear(in_features=3072, out_features=3072, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=3072, out_features=3072, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
        (norm_added_q): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (norm_added_k): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (add_q_proj): Linear(in_features=3072, out_features=3072, bias=True)
        (add_k_proj): Linear(in_features=3072, out_features=3072, bias=True)
        (add_v_proj): Linear(in_features=3072, out_features=3072, bias=True)
        (to_add_out): Linear(in_features=3072, out_features=3072, bias=True)
      )
      (norm2): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (ff): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): Linear(in_features=3072, out_features=12288, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=12288, out_features=3072, bias=True)
        )
      )
      (norm2_context): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (ff_context): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): Linear(in_features=3072, out_features=12288, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=12288, out_features=3072, bias=True)
        )
      )
    )
  )
  (single_transformer_blocks): ModuleList(
    (0-37): 38 x FluxSingleTransformerBlock(
      (norm): AdaLayerNormZeroSingle(
        (silu): SiLU()
        (linear): Linear(in_features=3072, out_features=9216, bias=True)
        (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      )
      (proj_mlp): Linear(in_features=3072, out_features=12288, bias=True)
      (act_mlp): GELU(approximate='tanh')
      (proj_out): Linear(in_features=15360, out_features=3072, bias=True)
      (attn): FluxAttention(
        (norm_q): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (norm_k): RMSNorm((128,), eps=1e-06, elementwise_affine=True)
        (to_q): Linear(in_features=3072, out_features=3072, bias=True)
        (to_k): Linear(in_features=3072, out_features=3072, bias=True)
        (to_v): Linear(in_features=3072, out_features=3072, bias=True)
      )
    )
  )
  (norm_out): AdaLayerNormContinuous(
    (silu): SiLU()
    (linear): Linear(in_features=3072, out_features=6144, bias=True)
    (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
  )
  (proj_out): Linear(in_features=3072, out_features=64, bias=True)
)
```

### Corresponding Pattern Dictionary

Based on the model structure above, the `pattern_dict` for FLUX.1 is given as follows. It is used as an argument to `dfloat11.compress_model` to specify the compression configuration.

```python
{
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
}
```
