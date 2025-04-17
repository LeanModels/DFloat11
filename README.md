# DFloat11: Lossless LLM Compression for Efficient GPU Inference

**DFloat11** is a lossless compression framework that reduces the size of Large Language Models (LLMs) by approximately **30%** while preserving **bit-for-bit identical outputs** to the original model. It enables efficient GPU inference on resource-constrained hardware without sacrificing accuracy.

## 📦 Installation

Requires CUDA-compatible GPU, and [PyTorch](https://pytorch.org/get-started/locally/) installed.

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

Run inference with a DFloat11-compressed LLM:

### Example Command

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --df11_name_or_path DFloat11/Llama-3.1-8B-Instruct-DF11 \
  --prompt "Question: What is a binary tree and its applications? Answer:" \
  --num_tokens 512 \
  --batch_size 1
```

> 💡 **Tip**: If you specify multiple CUDA devices (e.g., `CUDA_VISIBLE_DEVICES=0,1`), the model will be automatically distributed across them using `device_map="auto"`.

### Arguments

- `--model_name_or_path`: HuggingFace model name or local path (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- `--df11_name_or_path`: Path or repo for the corresponding DFloat11-compressed model (e.g., `DFloat11/Llama-3.1-8B-Instruct-DF11`)
- `--use_bf16`: *(Optional)* Load the original BFloat16 model instead of the compressed one
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

### Model Usage

To use a DFloat11-compressed LLM like a standard HuggingFace model:

```python
from dfloat11 import DFloat11ModelForCausalLM

model = DFloat11ModelForCausalLM.from_pretrained(
    "<huggingface-model-name>",
    "<path-to-dfloat11-model>",
    device_map="auto",
)

# The model behaves like a regular HuggingFace CausalLM
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
