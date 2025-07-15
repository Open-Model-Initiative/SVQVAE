import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from safetensors.torch import load_file
from dataclasses import dataclass, field
from modules import Encoder, Decoder, VectorQuantizer


# Utils
def _make_window(size: int, device, dtype) -> torch.Tensor:
    w1d = torch.hann_window(size, periodic=False, device=device, dtype=torch.float32)
    return torch.outer(w1d, w1d).unsqueeze(0).unsqueeze(0)


def _tile_grid(length: int, tile: int, overlap: int) -> torch.Tensor:
    stride = tile - overlap
    if length <= tile:
        return torch.tensor([0])
    num = math.ceil((length - overlap) / stride)
    coords = torch.arange(num) * stride
    last = length - tile
    if coords[-1] != last:
        coords[-1] = last
    return coords


def _num_tiles(length: int, tile: int, overlap: int) -> int:
    return _tile_grid(length, tile, overlap).numel()


def _blend_onto_latent(buf, wbuf, tile, win, y0, x0):
    t = win.shape[-1]
    buf[:, :, y0:y0+t, x0:x0+t].add_(tile * win)
    wbuf[:, :, y0:y0+t, x0:x0+t].add_(win)


def _blend_onto_pixel(buf, tile, win, y0, x0):
    h, w = win.shape[-2:]
    buf[:, 0:1, y0:y0+h, x0:x0+w].add_(win)
    buf[:, 1:, y0:y0+h, x0:x0+w].add_(tile * win)


# ModelArgs
@dataclass
class ModelArgs:
    codebook_size: int = 16_384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


# Model
class ScalableVQVAE(nn.Module):
    def __init__(self, cfg: ModelArgs):
        super().__init__()
        self.cfg = cfg

        # encoder / decoder
        self.encoder = Encoder(
            ch_mult=cfg.encoder_ch_mult,
            z_channels=cfg.z_channels,
            dropout=cfg.dropout_p,
        )
        self.decoder = Decoder(
            z_channels=cfg.z_channels,
            ch_mult=cfg.decoder_ch_mult,
            dropout=cfg.dropout_p,
        )

        # vector‑quantiser + 1×1 projections
        self.quantize = VectorQuantizer(
            cfg.codebook_size,
            cfg.codebook_embed_dim,
            cfg.commit_loss_beta,
            cfg.entropy_loss_ratio,
            cfg.codebook_l2_norm,
            cfg.codebook_show_usage,
        )
        self.quant_conv = nn.Conv2d(cfg.z_channels, cfg.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(cfg.codebook_embed_dim, cfg.z_channels, 1)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256, device=self.quant_conv.weight.device)
            self.ds = 256 // self.encoder(dummy).shape[-1]

        self.mean = torch.Tensor([0.5351199507713318,0.5217390656471252,0.4665282666683197])
        self.std = torch.Tensor([0.28029370307922363,0.27318093180656433,0.28354412317276])

    # Helpers
    def encode(self, x):
        h = self.quant_conv(self.encoder(x))
        return self.quantize(h)

    def decode(self, quant):
        return self.decoder(self.post_quant_conv(quant))

    # Tile Encode
    @torch.inference_mode()
    def _encode_tiled(self, x, tile_px: int = 256, overlap_px: int = 32):
        B, _, H, W = x.shape
        stride_px  = tile_px - overlap_px
        pad_y = max(0, math.ceil((H - overlap_px)/stride_px)*stride_px + overlap_px - H)
        pad_x = max(0, math.ceil((W - overlap_px)/stride_px)*stride_px + overlap_px - W)
        x_pad = F.pad(x, (0, pad_x, 0, pad_y), mode="reflect")
        H_pad, W_pad = x_pad.shape[-2:]

        tile_lat = tile_px // self.ds
        Hl, Wl = H_pad // self.ds, W_pad // self.ds

        buf  = torch.zeros(B, self.cfg.codebook_embed_dim, Hl, Wl, device=x.device, dtype=torch.float32)
        wbuf = torch.zeros(1, 1, Hl, Wl, device=x.device, dtype=torch.float32)
        win  = _make_window(tile_lat, x.device, torch.float32)

        ys = _tile_grid(H_pad, tile_px, overlap_px)
        xs = _tile_grid(W_pad, tile_px, overlap_px)

        for y0 in ys:
            for x0 in xs:
                tile = x_pad[:, :, y0:y0+tile_px, x0:x0+tile_px]
                lat  = self.quant_conv(self.encoder(tile)).float()
                _blend_onto_latent(buf, wbuf, lat, win, y0//self.ds, x0//self.ds)

        buf.div_(wbuf + 1e-6)
        return buf, H_pad, W_pad

    # Latent Helper
    def _latent_target(self, H_pad, W_pad, tile_px, overlap_px, p_lat):
        tiles_y = _num_tiles(H_pad, tile_px, overlap_px)
        tiles_x = _num_tiles(W_pad, tile_px, overlap_px)
        overlap_lat = max(1, round((overlap_px//self.ds) * p_lat / (tile_px//self.ds)))
        stride_lat  = p_lat - overlap_lat
        tgt_h = (tiles_y - 1)*stride_lat + p_lat
        tgt_w = (tiles_x - 1)*stride_lat + p_lat
        return tgt_h, tgt_w, overlap_lat

    # Tile Decode
    def _decode_from_latent(
        self,
        latent: torch.Tensor,
        pix_buf: torch.Tensor,
        win_cache: Dict[int, torch.Tensor],
        H_pad: int,
        W_pad: int,
        p_lat: int,
        overlap_lat: int,
    ) -> torch.Tensor:
        tile_px = p_lat * self.ds

        if tile_px not in win_cache:
            win_cache[tile_px] = _make_window(tile_px, latent.device, torch.float32)
        win_px = win_cache[tile_px]

        pix_buf.zero_()

        ys = _tile_grid(latent.shape[-2], p_lat, overlap_lat)
        xs = _tile_grid(latent.shape[-1], p_lat, overlap_lat)

        for y0 in ys:
            for x0 in xs:
                lt = latent[:, :, y0:y0+p_lat, x0:x0+p_lat]
                qt, _, _ = self.quantize(lt)
                px = self.decode(qt).float()
                _blend_onto_pixel(pix_buf, px, win_px, y0*self.ds, x0*self.ds)

        rgb = pix_buf[:, 1:, :H_pad, :W_pad] / (pix_buf[:, 0:1, :H_pad, :W_pad] + 1e-6)
        return rgb.clamp_(0.0, 1.0)

    # Forward
    @torch.no_grad()
    def forward(self, x, patch_nums=None, tile_size=256, overlap=32, align: bool = False):
        if align:
            mu_new = x.mean(dim=(0,2,3), keepdim=True)
            sigma_new = x.std(dim=(0,2,3), unbiased=False, keepdim=True)
            eps = 1e-6

            train_mean = self.mean.to(x.device).view(1, -1, 1, 1)
            train_std = self.std.to(x.device).view(1, -1, 1, 1)

            x_al = (x - mu_new) / (sigma_new + eps)
            x_al = x_al * train_std + train_mean
        else:
            x_al = x

        if patch_nums is None:
            q, diff, _ = self.encode(x_al)
            dec = self.decode(q)

            if align:
                recon = (dec - train_mean) / (train_std + eps)
                recon = recon * sigma_new + mu_new
                return recon.clamp(0,1), diff
            else:
                return dec.clamp(0,1), diff

        patch_nums = sorted({int(p) for p in patch_nums})
        lat_buf, H_pad, W_pad = self._encode_tiled(x_al, tile_size, overlap)
        full_p   = tile_size // self.ds
        max_tile = max(patch_nums) * self.ds

        pix_buf = torch.empty(
            x.size(0), 4,
            H_pad + max_tile, W_pad + max_tile,
            device=x.device, dtype=torch.float32
        )
        win_cache: Dict[int, torch.Tensor] = {}

        outs = []
        for p in patch_nums:
            tgt_h, tgt_w, ov_lat = self._latent_target(H_pad, W_pad, tile_size, overlap, p)
            latent_scaled = F.interpolate(
                lat_buf, (tgt_h, tgt_w),
                mode="area" if p <= full_p else "bilinear"
            )

            img_al = self._decode_from_latent(
                latent_scaled, pix_buf, win_cache,
                H_pad, W_pad, p, ov_lat
            )

            if align:
                img = (img_al - train_mean) / (train_std + eps)
                img = img * sigma_new + mu_new
            else:
                img = img_al

            scale = p / full_p
            H, W = x.shape[-2:]
            out = img[:, :, :int(round(scale*H)), :int(round(scale*W))]
            outs.append(out.cpu().clamp(0,1))

        return outs, [None] * len(outs)

    # Load Func
    @staticmethod
    def load(
        safetensor_path: str,
        model_args: ModelArgs = ModelArgs(
            encoder_ch_mult=[1, 2, 2, 4],
            decoder_ch_mult=[1, 2, 2, 4],
            z_channels=32,
        ),
    ) -> "ScalableVQVAE":
        model = ScalableVQVAE(cfg=model_args)
        model.load_state_dict(load_file(safetensor_path, device="cpu"), strict=True)
        model.eval()
        return model