# 🖼️ Example: Text-to-Image with DFloat11 + FLUX.1-dev

FLUX.1-dev is a 12-billion parameter diffusion transformer model in **BFloat16**, requiring roughly **24GB of GPU memory**. Using **DFloat11**, the model is **losslessly compressed** to just **16.3GB**, with **bit-for-bit identical outputs** to the original.

👉 **DFloat11 introduces negligible latency overhead**---just a few extra seconds per image generation---while significantly reducing memory usage.

---

### ✅ Requirements

* A CUDA-compatible GPU with **at least 20GB of VRAM**
* Access to the [FLUX.1-dev model](https://huggingface.co/black-forest-labs/FLUX.1-dev)
* [PyTorch](https://pytorch.org/get-started/locally/) installed
* `diffusers` installed:

  ```bash
  pip install -U diffusers
  ```
* `dfloat11` installed:

  ```bash
  pip install -U dfloat11[cuda12]
  # For CUDA 11 users:
  # pip install -U dfloat11[cuda11]
  ```

---

### 🚀 Generate an Image with FLUX.1-dev

Use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python image_gen.py \
  --prompt "A futuristic cityscape at sunset, with flying cars, neon lights, and reflective water canals" \
  --save_path ./image.png
```

This will generate an image using the **compressed FLUX.1-dev model**, with **no loss in quality** and **minimal latency overhead**.
