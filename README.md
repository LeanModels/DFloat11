# DFloat11: Lossless LLM Compression for Efficient GPU Inference

[![PyPI version](https://img.shields.io/pypi/v/dfloat11.svg?color=blue)](https://pypi.org/project/dfloat11/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.11651-b31b1b.svg)](https://arxiv.org/abs/2504.11651)

**DFloat11** is a lossless compression framework that reduces the size of Large Language Models (LLMs) by approximately **30%** while preserving **bit-for-bit identical outputs** to the original model. It enables efficient GPU inference on resource-constrained hardware without sacrificing accuracy.

## 📦 Installation

Requires a CUDA-compatible GPU and [PyTorch](https://pytorch.org/get-started/locally/) installed.

```bash
pip install dfloat11[cuda12]
# or if you have CUDA version 11:
# pip install dfloat11[cuda11]
```

## 🔧 Key Features

- **📉 Significant size reduction**: Compresses LLM weights by ~30%, losslessly.
- **✅ Zero loss in accuracy**: Produces **bit-for-bit identical outputs** to the original BFloat16 model.
- **🧩 Easy to use**: Seamlessly integrates with HuggingFace framework.
- **⚡ High throughput**: Enables up to **38.8× faster** generation compared to CPU offloading alternatives.
- **🧠 Supports longer inputs**: Extends maximum context length by up to **13.17×** under the same GPU memory budget.

## 🔗 Links

👉 Explore pre-compressed DFloat11 models ready to use on HuggingFace: **[https://huggingface.co/DFloat11](https://huggingface.co/DFloat11)**

📂 Official Code Repository: [https://github.com/LeanModels/DFloat11](https://github.com/LeanModels/DFloat11)

## 🚀 Quick Start

To run inference with a DFloat11-compressed LLM:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --df11_name_or_path DFloat11/Llama-3.1-8B-Instruct-DF11 \
  --prompt "Question: What is a binary tree and its applications? Answer:" \
  --num_tokens 512 \
  --batch_size 1
```

> 💡 **Tip**: If you specify multiple CUDA devices (e.g., `CUDA_VISIBLE_DEVICES=0,1`), the model will be automatically distributed across them using 🤗 Accelerate's `device_map="auto"`.

### Arguments

- `--model_name_or_path`: HuggingFace name or local path of the original BFloat16 model (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- `--df11_name_or_path`: HuggingFace name or local path of the corresponding DFloat11 model (e.g., `DFloat11/Llama-3.1-8B-Instruct-DF11`)
- `--use_bf16`: *(Optional)* Load the original BFloat16 model instead of the compressed one
- `--prompt`: Input prompt string for text generation
- `--num_tokens`: Number of new tokens to generate per sample
- `--batch_size`: Number of prompts to process in parallel
- `--seed`: *(Optional)* Random seed for reproducible results

See the [Model Hub](#model-hub) section for a list of available DFloat11 models.

### Output

The script prints:
- Generated responses
- Total decoding latency
- Tokens per second (throughput)
- GPU memory usage (allocated and peak)

## Model Hub

| dfloat11-model-name | bfloat16-model-name |
|---------------------------|-----------------------------|
| [DFloat11/Llama-3.1-405B-Instruct-DF11](https://huggingface.co/DFloat11/Llama-3.1-405B-Instruct-DF11) | [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) |
| [DFloat11/Llama-3.1-8B-Instruct-DF11](https://huggingface.co/DFloat11/Llama-3.1-8B-Instruct-DF11) | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| [DFloat11/Llama-3.3-70B-Instruct-DF11](https://huggingface.co/DFloat11/Llama-3.3-70B-Instruct-DF11) | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| [DFloat11/gemma-3-12b-it-DF11](https://huggingface.co/DFloat11/gemma-3-12b-it-DF11) | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) |
| [DFloat11/gemma-3-27b-it-DF11](https://huggingface.co/DFloat11/gemma-3-27b-it-DF11) | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) |
| [DFloat11/Mistral-Nemo-Instruct-2407-DF11](https://huggingface.co/DFloat11/Mistral-Nemo-Instruct-2407-DF11) | [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| [DFloat11/Mistral-Small-24B-Instruct-2501-DF11](https://huggingface.co/DFloat11/Mistral-Small-24B-Instruct-2501-DF11) | [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) |
| [DFloat11/Qwen2.5-14B-Instruct-DF11](https://huggingface.co/DFloat11/Qwen2.5-14B-Instruct-DF11) | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| [DFloat11/Qwen2.5-32B-Instruct-DF11](https://huggingface.co/DFloat11/Qwen2.5-32B-Instruct-DF11) | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| [DFloat11/QwQ-32B-DF11](https://huggingface.co/DFloat11/QwQ-32B-DF11) | [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| [DFloat11/DeepSeek-R1-Distill-Llama-8B-DF11](https://huggingface.co/DFloat11/DeepSeek-R1-Distill-Llama-8B-DF11) | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| [DFloat11/DeepSeek-R1-Distill-Qwen-7B-DF11](https://huggingface.co/DFloat11/DeepSeek-R1-Distill-Qwen-7B-DF11) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| [Find more models on our HF page!](https://huggingface.co/DFloat11) | ... |

### How to Use a DFloat11 Model

1. Download a model using the Hugging Face command line tool:
```bash
huggingface-cli download \
  DFloat11/DeepSeek-R1-Distill-Qwen-7B-DF11 \     # DFloat11 model name
  --local-dir ./DeepSeek-R1-Distill-Qwen-7B-DF11  # local path to download the DFloat11 model
```
2. Use the model like a standard Hugging Face model:
```python
from dfloat11 import DFloat11ModelForCausalLM

model = DFloat11ModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # original BFloat16 model name
    "./DeepSeek-R1-Distill-Qwen-7B-DF11",         # local path to DFloat11 model
    device_map="auto",
)
```

## 🧠 Contributions

This work is brought to you by the team at Rice University and [xMAD.ai](https://xmad.ai/).

The GPU kernel was designed and implemented by [Tianyi Zhang](https://github.com/tonyzhang617).

## 📚 Citation

If you found our work useful or interesting, please consider citing our paper:

```bibtex
@misc{zhang2025dfloat11,
  title        = {70\% Size, 100\% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float},
  author       = {Tianyi Zhang and Yang Sui and Shaochen Zhong and Vipin Chaudhary and Xia Hu and Anshumali Shrivastava},
  year         = {2025},
  eprint       = {2504.11651},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.11651}
}
```
