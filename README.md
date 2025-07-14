# SVQVAE (Scalable Vector Quantized Variational Autoencoder)

A PyTorch implementation of a scalable Vector Quantized Variational Autoencoder (VQVAE) for high-resolution image generation and reconstruction. This implementation supports tiled processing for handling large images efficiently.

## Overview

SVQVAE is a scalable variant of the Vector Quantized Variational Autoencoder that can process high-resolution images through tiled encoding and decoding. The model uses a discrete codebook to compress images into a latent representation and can reconstruct them at multiple scales.

### Key Features

- **Scalable Processing**: Handles high-resolution images through tiled processing
- **Multi-scale Output**: Can generate reconstructions at different scales
- **Vector Quantization**: Uses a discrete codebook for efficient compression
- **Attention Mechanisms**: Includes self-attention blocks for better feature learning
- **Flexible Architecture**: Configurable encoder/decoder with customizable channel multipliers

## Architecture

The model consists of three main components:

1. **Encoder**: Convolutional network that compresses input images to latent representations
2. **Vector Quantizer**: Discretizes continuous latent vectors using a learned codebook
3. **Decoder**: Reconstructs images from quantized latent representations

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SVQVAE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import torch
from sqvae import ScalableVQVAE, ModelArgs

# Initialize model with default configuration
model = ScalableVQVAE(ModelArgs())

# Load pre-trained weights (if available)
# model = ScalableVQVAE.load("path/to/model.safetensors")

# Prepare input image (B, C, H, W)
x = torch.randn(1, 3, 256, 256)

# Forward pass
with torch.no_grad():
    reconstructed, diff = model(x)
```

### Tiled Processing for Large Images

```python
# Process large images with tiling
x = torch.randn(1, 3, 1024, 1024)

# Generate reconstructions at multiple scales
with torch.no_grad():
    outputs, diffs = model(x, patch_nums=[1, 2, 4], tile_size=256, overlap=32)
    
# outputs[0] - 1x scale reconstruction
# outputs[1] - 2x scale reconstruction  
# outputs[2] - 4x scale reconstruction
```

### Model Configuration

```python
from sqvae import ModelArgs

# Custom configuration
config = ModelArgs(
    codebook_size=16384,           # Number of codebook entries
    codebook_embed_dim=8,          # Embedding dimension
    encoder_ch_mult=[1, 1, 2, 2, 4],  # Encoder channel multipliers
    decoder_ch_mult=[1, 1, 2, 2, 4],  # Decoder channel multipliers
    z_channels=256,                # Latent channels
    commit_loss_beta=0.25,         # Commitment loss weight
    dropout_p=0.0                  # Dropout probability
)

model = ScalableVQVAE(config)
```

## Model Parameters

### ModelArgs Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `codebook_size` | 16384 | Number of discrete codebook entries |
| `codebook_embed_dim` | 8 | Dimension of codebook embeddings |
| `codebook_l2_norm` | True | Whether to L2 normalize embeddings |
| `commit_loss_beta` | 0.25 | Weight for commitment loss |
| `encoder_ch_mult` | [1,1,2,2,4] | Channel multipliers for encoder |
| `decoder_ch_mult` | [1,1,2,2,4] | Channel multipliers for decoder |
| `z_channels` | 256 | Number of latent channels |
| `dropout_p` | 0.0 | Dropout probability |

## Training

The model can be trained using standard VQVAE training procedures:

1. **Reconstruction Loss**: MSE between input and reconstructed images
2. **Commitment Loss**: Ensures encoder outputs stay close to codebook
3. **Codebook Loss**: Updates codebook embeddings

### Training Example

```python
# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()
    
    reconstructed, (vq_loss, commit_loss, entropy_loss, usage) = model(batch)
    
    # Compute losses
    recon_loss = F.mse_loss(reconstructed, batch)
    total_loss = recon_loss + vq_loss + commit_loss
    
    total_loss.backward()
    optimizer.step()
```

## Inference

### Single Image Processing

```python
# Load and preprocess image
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

image = Image.open("input.jpg")
x = transform(image).unsqueeze(0)

# Reconstruct
with torch.no_grad():
    reconstructed, _ = model(x)
```

### Multi-scale Generation

```python
# Generate at multiple scales
with torch.no_grad():
    outputs, _ = model(x, patch_nums=[1, 2, 4], tile_size=256, overlap=32)
    
    # Save results
    for i, output in enumerate(outputs):
        save_image(output, f"reconstruction_{2**i}x.png")
```

## Examples

See `inference_example.ipynb` for detailed usage examples and visualizations.

## Dependencies

- PyTorch >= 2.0.0
- torchvision
- safetensors
- numpy
- matplotlib
- pillow

## License

The code in this repository is licensed under the Apache 2.0 license.

Model weights

## Citation

If you use this code in your research, please cite Austin J. Bryant and the Open Model Initiative.

## Acknowledgments

This implementation is based on the VQVAE architecture and includes improvements for scalable processing of high-resolution images.