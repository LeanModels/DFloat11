# 🧪 Example: Text-to-Video with DFloat11 + Wan2.1

[Wan2.1](https://github.com/Wan-Video/Wan2.1) is a diffusion transformer for high-quality video generation. With DFloat11, a lossless compression format designed for GPU inference, the model is compressed to \~70% of its original size without any degradation in output quality.

DFloat11 enables efficient video generation with Wan2.1-14B (28GB) **on a single 24GB GPU**.

---

## 📊 Performance Comparison

| Metric                             | Wan2.1-T2V-14B (BFloat16) | Wan2.1-T2V-14B (DFloat11) |
| ---------------------------------- | ------------------------- | ------------------------- |
| Model Size                         | 28.64 GB                  | 19.39 GB                  |
| Peak GPU Memory<br>(2s 480p Video) | 30.79 GB                  | 22.22 GB                  |
| Generation Time<br>(an A100 GPU)   | 339 seconds               | 348 seconds               |

---

## ✅ Requirements

* CUDA-compatible GPU with **≥ 24GB VRAM**
* Install Wan2.1 and DFloat11 dependencies:

```bash
# Install Wan2.1 dependencies
pip install -r requirements.txt

# Install DFloat11
pip install -U dfloat11[cuda12]
# For CUDA 11 users:
# pip install -U dfloat11[cuda11]
```

---

## 🚀 Generate Video from Text Prompt

Run the following command to generate a video using the **DFloat11-compressed Wan2.1-14B model**:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_video.py \
    --prompt "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window." \
    --output_file_name output.mp4
```

This script automatically:

* Downloads the Wan2.1-14B video generation pipeline
* Downloads and applies **DFloat11 lossless compression** to the model weights
* Runs efficient text-to-video generation on `cuda:0`
* Generates a video with **reduced memory usage**, **no quality loss**, and saves it as `output.mp4`
