# DFloat11: Lossless LLM Compression for Efficient GPU Inference

[![PyPI version](https://img.shields.io/pypi/v/dfloat11.svg?color=blue)](https://pypi.org/project/dfloat11/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.11651-b31b1b.svg)](https://arxiv.org/abs/2504.11651)
[![Hugging Face](https://img.shields.io/badge/Model-%F0%9F%A4%97-yellow.svg)](https://huggingface.co/DFloat11)

**DFloat11** is a lossless compression framework that reduces the size of Large Language Models (LLMs) by approximately **30%** while preserving **bit-for-bit identical outputs** to the original model. It enables efficient GPU inference on resource-constrained hardware without sacrificing accuracy.

## 📰 News
- [07/29/2025] Efficient CPU Offloading Now Supported!
  * Our latest update enables highly memory-efficient inference by keeping only one transformer block in GPU memory at a time. For example, CPU offloading reduces peak GPU memory for inference of **FLUX.1-Krea-dev from 17.5 to 9.8 GB, Qwen3-8B from 12.4 to 2.3 GB, and HiDream-I1-Full from 26.4 to 9.6 GB**.
  * An example of using CPU offloading with FLUX.1-Krea-dev-DF11 can be found [here](https://huggingface.co/DFloat11/FLUX.1-Krea-dev-DF11).
  * To enable CPU offloading, simply set `cpu_offload=True` when calling `DFloat11Model.from_pretrained(...)`.
- [05/23/2025] **Wan2.1** support is now live! [`DFloat11/Wan2.1-T2V-14B-Diffusers-DF11`](https://huggingface.co/DFloat11/Wan2.1-T2V-14B-Diffusers-DF11)
  * Text-to-video generation with DFloat11 *Wan2.1 14B* using only 24GB VRAM!
  * Get started here: [examples/wan2.1](https://github.com/LeanModels/DFloat11/tree/master/examples/wan2.1).
- [05/06/2025] **DFloat11 now supports [`FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev)**
  * 🖼️ Generate stunning text-to-image results on GPUs with **less than 24GB VRAM** --- no quality lost!
  * 📂 Get started here: [examples/flux.1](https://github.com/LeanModels/DFloat11/tree/master/examples/flux.1).
- [05/05/2025] The `dfloat11` pip package has been upgraded to `v0.2.0`! Run `pip install -U dfloat11[cuda12]` to upgrade to the latest version. We have made the following important changes:
  * We added support for Qwen 3, Gemma 3, and Phi 4!
  * The GPU decompression kernel is now 20-40% faster! We achieved it by improving thread occupancy and implementing tons of optimizations.
  * The DFloat11 models are now stored in safetensors format for better safety and loading performance.
  * When using a DFloat11 model, only the compressed model is downloaded, not the original model.

## 📦 Installation

Requires a CUDA-compatible GPU and [PyTorch](https://pytorch.org/get-started/locally/) installed.

```bash
pip install -U dfloat11[cuda12]
# or if you have CUDA version 11:
# pip install -U dfloat11[cuda11]
```

## 🔍 How It Works

DFloat11 compresses model weights using **Huffman coding** of BFloat16 exponent bits, combined with **hardware-aware algorithmic designs** that enable efficient on-the-fly decompression directly on the GPU. During inference, the weights remain compressed in GPU memory and are **decompressed just before matrix multiplications**, then **immediately discarded after use** to minimize memory footprint.

Key benefits:

* **No CPU decompression or host-device data transfer**: all operations are handled entirely on the GPU.
* **Decompression overhead is constant** per forward pass and **independent of batch size**, making DFloat11 increasingly efficient at larger batch sizes.
* DFloat11 is **much faster than CPU-offloading approaches**, enabling practical deployment in memory-constrained environments.
* At batch size = 1, inference is approximately 2× slower than the original BF16 model, but the performance gap narrows significantly with larger batches.
* The compression is **fully lossless**, guaranteeing that the model’s outputs are **bit-for-bit identical** to those of the original model.

## 🚀 Quick Start

1. Install the `dfloat11` pip package. See [Installation](#-installation).
2. Run the following code in Python, which automatically downloads the DFloat11 `Qwen3-8B` model and generates a response.
  ```python
  import torch
  from dfloat11 import DFloat11Model
  from transformers import AutoTokenizer

  model_id = "DFloat11/Qwen3-8B-DF11"

  model = DFloat11Model.from_pretrained(model_id, device_map="auto")

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token

  prompt = "Question: What is a binary tree and its applications? Answer:"
  inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

  with torch.no_grad():
      output = model.generate(
          **inputs,
          max_new_tokens=256,
          do_sample=True,
      )

  print(tokenizer.batch_decode(output, skip_special_tokens=True))
  ```
3. Replace the `model_id` in the script above with any pre-compressed model in the [Model Hub](#-model-hub).

## 🏎️ Benchmarking Performance

To test the speed and memory consumption a DFloat11 LLM during inference:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_name_or_path DFloat11/Qwen3-8B-DF11 \
  --prompt "Question: What is a binary tree and its applications? Answer:" \
  --num_tokens 512 \
  --batch_size 1
```

> 💡 **Tip**: If you specify multiple CUDA devices (e.g., `CUDA_VISIBLE_DEVICES=0,1`), the model will be automatically distributed across them using 🤗 Accelerate's `device_map="auto"`.

### Arguments

- `--model_name_or_path`: HuggingFace name or local path of the DFloat11 model (e.g., `DFloat11/Qwen3-8B-DF11`). See the [Model Hub](#-model-hub) section for a list of available DFloat11 models.
- `--bf16`: *(Optional)* Turn on this flag when passing a BFloat16 model to `--model_name_or_path`
- `--prompt`: Input prompt string for text generation
- `--num_tokens`: Number of new tokens to generate per sample
- `--batch_size`: Number of prompts to process in parallel
- `--seed`: *(Optional)* Random seed for reproducible results

### Output

The script prints:
- Generated responses
- Total decoding latency
- Tokens per second (throughput)
- GPU memory usage (allocated and peak)

## 📚 Model Hub

| Model | DFloat11 Link |
|-------|---------------|
| Wan2.1 T2V 14B (see [examples/wan2.1](https://github.com/LeanModels/DFloat11/tree/master/examples/wan2.1)) | [DFloat11/Wan2.1-T2V-14B-Diffusers-DF11](https://huggingface.co/DFloat11/Wan2.1-T2V-14B-Diffusers-DF11) |
| FLUX.1 dev (see [examples/flux.1](https://github.com/LeanModels/DFloat11/tree/master/examples/flux.1)) | [DFloat11/FLUX.1-dev-DF11](https://huggingface.co/DFloat11/FLUX.1-dev-DF11) |
| Qwen 3 32B | [DFloat11/Qwen3-32B-DF11](https://huggingface.co/DFloat11/Qwen3-32B-DF11) |
| Qwen 3 14B | [DFloat11/Qwen3-14B-DF11](https://huggingface.co/DFloat11/Qwen3-14B-DF11) |
| Qwen 3 8B | [DFloat11/Qwen3-8B-DF11](https://huggingface.co/DFloat11/Qwen3-8B-DF11) |
| Qwen 3 4B | [DFloat11/Qwen3-4B-DF11](https://huggingface.co/DFloat11/Qwen3-4B-DF11) |
| Phi 4 Reasoning Plus | [DFloat11/Phi-4-reasoning-plus-DF11](https://huggingface.co/DFloat11/Phi-4-reasoning-plus-DF11) |
| Gemma 3 27B Instruct | [DFloat11/gemma-3-27b-it-DF11](https://huggingface.co/DFloat11/gemma-3-27b-it-DF11) |
| Gemma 3 12B Instruct | [DFloat11/gemma-3-12b-it-DF11](https://huggingface.co/DFloat11/gemma-3-12b-it-DF11) |
| Gemma 3 4B Instruct  | [DFloat11/gemma-3-4b-it-DF11](https://huggingface.co/DFloat11/gemma-3-4b-it-DF11) |
| Llama 3.1 8B Instruct | [DFloat11/Llama-3.1-8B-Instruct-DF11](https://huggingface.co/DFloat11/Llama-3.1-8B-Instruct-DF11) |
| DeepSeek R1 Distill Qwen 32B | [DFloat11/DeepSeek-R1-Distill-Qwen-32B-DF11](https://huggingface.co/DFloat11/DeepSeek-R1-Distill-Qwen-32B-DF11) |
| DeepSeek R1 Distill Qwen 14B | [DFloat11/DeepSeek-R1-Distill-Qwen-14B-DF11](https://huggingface.co/DFloat11/DeepSeek-R1-Distill-Qwen-14B-DF11) |
| DeepSeek R1 Distill Qwen 7B  | [DFloat11/DeepSeek-R1-Distill-Qwen-7B-DF11](https://huggingface.co/DFloat11/DeepSeek-R1-Distill-Qwen-7B-DF11) |
| DeepSeek R1 Distill Llama 8B | [DFloat11/DeepSeek-R1-Distill-Llama-8B-DF11](https://huggingface.co/DFloat11/DeepSeek-R1-Distill-Llama-8B-DF11) |
| ... | [Discover more models on our HF page!](https://huggingface.co/DFloat11) |

### How to Use a DFloat11 Model

1. Download a model using the HuggingFace command line tool:
  ```bash
  huggingface-cli download \
    DFloat11/Llama-3.1-8B-Instruct-DF11 \     # DFloat11 model name
    --local-dir ./Llama-3.1-8B-Instruct-DF11  # local path to download the DFloat11 model
  ```
2. Run the following in Python to load the model and tokenizer:
  ```python
  from dfloat11 import DFloat11Model
  from transformers import AutoTokenizer

  model_path = "./Llama-3.1-8B-Instruct-DF11"
  model = DFloat11Model.from_pretrained(model_path, device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  ```

## 🔗 Links

👉 Explore pre-compressed DFloat11 models ready to use on HuggingFace: **[https://huggingface.co/DFloat11](https://huggingface.co/DFloat11)**

📂 Official Code Repository: [https://github.com/LeanModels/DFloat11](https://github.com/LeanModels/DFloat11)

## 🧠 Contributions

This work is brought to you by the team at Rice University and [xMAD.ai](https://xmad.ai/).

The GPU kernel was designed and implemented by [Tianyi Zhang](https://github.com/tonyzhang617).

## 📚 Citation

If you found our work useful or interesting, please consider citing our paper:

```bibtex
@article{zhang2025dfloat11,
  title={70\% Size, 100\% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float},
  author={Zhang, Tianyi and Sui, Yang and Zhong, Shaochen and Chaudhary, Vipin and Hu, Xia and Shrivastava, Anshumali},
  journal={arXiv preprint arXiv:2504.11651},
  year={2025}
}
```
