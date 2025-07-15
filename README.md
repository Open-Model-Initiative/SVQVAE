# Scalable VQ‑VAE (SVQVAE)

[![Licence](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#installation)
[![Paper](https://img.shields.io/badge/arXiv-2502.20313-b31b1b.svg)](https://arxiv.org/abs/2502.20313)

A **scalable Vector‑Quantised Variational Autoencoder (VQ‑VAE)** that supports *tile‑wise* encoding/decoding.

---

## ✨ Highlights

| Feature | Description |
|---------|-------------|
| **Scalable tiling** | Encode & decode arbitrarily large images by sliding window. |
| **Multi‑scale outputs** | Return reconstructions at user‑selected latent patch sizes (4 × 4 → 64 × 64). |
| **Lightweight codebook** | 16 k entries, 8‑D embeddings. |

## 🏗️ Architecture

    Input → Encoder → **Vector Quantiser** → Decoder → Output

* **Encoder / Decoder** – channel multiplier controlled by `encoder_ch_mult`, `decoder_ch_mult` (default `[1,2,2,4]`).  
* **Vector Quantiser** – codebook size 16 384, embed dim 8.   
For theory see **FlexVAR: Flexible Visual Autoregressive Modelling without Residual Prediction** (Jiao *et al.*, 2025).

---

## 🚀 Quick Start

### 1 · Install

```bash
git clone https://github.com/Open-Model-Initiative/SVQVAE.git
cd SVQVAE
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2 · Download a checkpoint

```bash
curl -L \
  https://huggingface.co/openmodelinitiative/SVQVAE/resolve/main/svqvae_weights.safetensors \
  -o svqvae_weights.safetensors
```

### 3 · Single‑image inference

```python
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from svqvae import ScalableVQVAE, ModelArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ScalableVQVAE.load(
    "svqvae_weights.safetensors",
    model_args=ModelArgs(
                codebook_size=16384,
                codebook_embed_dim=8,
                codebook_l2_norm=True,
                codebook_show_usage=True,
                commit_loss_beta=0.25,
                entropy_loss_ratio=0.0,
                encoder_ch_mult=[1, 2, 2, 4],
                decoder_ch_mult=[1, 2, 2, 4],
                z_channels=32,
                dropout_p=0.0,
            )
).to(device)

pil_img = Image.open("example.jpg").convert("RGB")
x = to_tensor(pil_img).unsqueeze(0).to(device)

scales = [4, 8, 16, 32, 64]   # latent patch sizes
with torch.no_grad():
    outs, _ = model(x, 
                    patch_nums=scales,
                    tile_size=512,   # RGB tile size
                    overlap=256)     # RGB tile overlap
```

See [`inference_example.ipynb`](inference_example.ipynb) for end‑to‑end demos.

---

## 🗿 Example notebook output for batched inference

<div align="center">
  <img src="output_final_2.png" width="200" alt="Output 2">
  <img src="output_final_3.png" width="200" alt="Output 3">
</div>

---

## 🏋️‍♂️ Training


Coming Soon!

---

## 📄 Citation

```bibtex
@misc{jiao2025flexvarflexiblevisualautoregressive,
      title={FlexVAR: Flexible Visual Autoregressive Modeling without Residual Prediction}, 
      author={Siyu Jiao and Gengwei Zhang and Yinlong Qian and Jiancheng Huang and Yao Zhao and Humphrey Shi and Lin Ma and Yunchao Wei and Zequn Jie},
      year={2025},
      eprint={2502.20313},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20313}, 
}

@software{bryant2025svqvae,
  author  = {Austin J. Bryant},
  title   = {SVQVAE: Scalable Vector Quantised Variational Autoencoder},
  url     = {https://github.com/Open-Model-Initiative/SVQVAE},
  version = {1.0.0},
  year    = {2025},
  license = {Apache‑2.0}
}
```

## 🔒 License

Released under the **Apache 2.0** license – see [`LICENSE`](LICENSE).
---

## 🙏 Acknowledgements

Huge thanks to the [OMI](https://openmodel.foundation/) community as a whole and to [Invoke](https://www.invoke.com/) for sponsoring the compute.