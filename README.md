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

## 🧪 Quickstart

```python
from dfloat11 import DFloat11ModelForCausalLM

model = DFloat11ModelForCausalLM.from_pretrained(
    "<huggingface-model-name>",
    "<path-to-dfloat11-model>",
    device_map='auto',
)

# model is ready to use like a regular huggingface model
```

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
