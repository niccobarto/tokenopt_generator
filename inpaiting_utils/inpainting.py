import torch
from torch import nn,Tensor
import enum
from tokenopt_generator.token_opt.tto.test_time_opt import (TestTimeOptConfig,TestTimeOpt,CLIPObjective,MultiObjective)
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
import torchvision.transforms.v2.functional as tvf
from PIL import Image
from einops import rearrange

#region Objectives
class ReconstructionObjective(nn.Module):
    """
    Penalizza le differenze solo FUORI dal buco (dove mask=1).
    Dentro al buco (mask=0) non agisce.
    Compatibile con MultiObjective.
    """

    def __init__(self, masked_img: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            masked_img: immagine originale mascherata (mask * img)
            mask: out-mask binaria (1 = fuori, 0 = dentro)
        """
        super().__init__()
        self.masked_img = masked_img
        self.mask = mask

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Assicuriamoci che masked_img e mask siano sullo stesso device di img
        dev = img.device
        if self.masked_img.device != dev:
            try:
                self.masked_img = self.masked_img.to(dev)
            except Exception:
                self.masked_img = self.masked_img
        if self.mask.device != dev:
            try:
                self.mask = self.mask.to(dev)
            except Exception:
                self.mask = self.mask

        # penalizza differenze solo dove mask=1 (fuori dal buco)
        diff = (img - self.masked_img).abs() * self.mask
        denom = self.mask.sum(dim=(1,2,3)) + 1e-8
        loss = diff.sum(dim=(1,2,3)) / denom
        return loss

class ComposedCLIP(nn.Module):
    """
    CLIP sul composito: FUORI = img originale, DENTRO = img ottimizzata.
    Gradiente: 1.0 dentro, outside_grad fuori (default 0.2).
    Niente buffer registrati: orig/mask restano su CPU per non occupare VRAM.
    """

    def __init__(self, base_clip_obj: nn.Module, orig_img: torch.Tensor,
                 mask_bin: torch.Tensor, outside_grad: float = 0.0):
        super().__init__()
        self.base = base_clip_obj
        self.orig = orig_img
        self.mask = mask_bin
        self.outside_grad = float(outside_grad)  # 0..1

    def get_prompt(self):
        return self.base.prompt

    def forward(self, img_opt: torch.Tensor) -> torch.Tensor:
        device = img_opt.device
        orig = self.orig.to(device, non_blocking=True)  # [1,3,H,W]
        mask = self.mask.to(device, non_blocking=True)  # [1,1,H,W], 1=FUORI, 0=BUCO

        # composito "visivo" (quello che CLIP vede)
        composed_vis = mask * orig + (1. - mask) * img_opt

        # maschera di gradiente: pieno dentro, attenuato fuori
        grad_mask = (1. - mask) + self.outside_grad * mask  # 1 dentro, outside_grad fuori

        # stesso tensore visivo, ma con gradiente pesato (trucco detach)
        composed_for_grad = grad_mask * img_opt + (1. - grad_mask) * img_opt.detach()
        composed = composed_vis.detach() + (composed_for_grad - composed_for_grad.detach())

        return self.base(composed)

class ObjectiveType(enum.Enum):
        ReconstructionObjective = 1
        ComposedCLIP = 2
        ComposedSiglip = 3
#endregion

#region Configurations

@dataclass
class Config:
    tto_config:TestTimeOptConfig #TestTimeOptConfig
    objective_weights:list[float] #lista di pesi peg li objectives
    cfg_scale:float
    num_augmentations:int
    two_phase_generation:bool=False
    enable_token_reset:bool=True
    reset_period:int=15 if enable_token_reset else None,
# ============================================================
# Helper per aggiungere configurazioni
# ============================================================
def add_conf(name, tto_params, cfg_scale, num_aug, weights,
             enable_token_reset, reset_period,objective_types):

    tto_config = TestTimeOptConfig(
        num_iter=tto_params["num_iter"],
        ema_decay=tto_params["ema_decay"],
        lr=tto_params["lr"],
        enable_amp=tto_params["enable_amp"],
        reg_type=tto_params["reg_type"],
        reg_weight=tto_params["reg_weight"],
        token_noise=tto_params["token_noise"],
    )

    conf = Config(
        tto_config=tto_config,
        objective_weights=weights,
        cfg_scale=cfg_scale,
        num_augmentations=num_aug,
        enable_token_reset=enable_token_reset,
        reset_period=reset_period,
    )
    return name, conf, objective_types

#endregion


#region Utility per caricamento immagini/maschere e conversione tensore<->immagine
def image_to_tensor(image_path: Path, size: int = 256, device = None) -> Tensor:
    img = Image.open(image_path).convert("RGB")
    t = (1.0 / 255.0) * torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1)
    # torchvision v2 expects size as sequence for some overloads; pass [size, size]
    t = tvf.resize(t, [size, size])
    t = tvf.center_crop(t, [size, size])
    t = t.unsqueeze(0)  # [1,3,H,W]
    if device is not None:
        t = t.to(device)
    return t


def mask_to_tensor(mask_path: Path, size: int = 256, threshold: float = 0.5, device = None) -> Tensor:
    """
    Carica maschera, la converte in scala di grigi, la ridimensiona e la binarizza.
    Restituisce tensore [1,1,H,W] con valori 0.0/1.0 (1 = FUORI, 0 = BUCO).
    Per default (device=None) mantiene il tensore su CPU per risparmiare VRAM;
    passare device=DEVICE per spostarlo subito su GPU.
    """
    m = Image.open(mask_path).convert("L")  # scala di grigi
    arr = (np.array(m).astype(np.float32) / 255.0)
    t = torch.from_numpy(arr)  # [H,W]
    t = t.unsqueeze(0)  # [1,H,W] per compatibilità con tvf
    # same resizing behaviour as images
    t = tvf.resize(t, [size, size])
    t = tvf.center_crop(t, [size, size])
    t = (t >= threshold).to(dtype=torch.float32)  # binarizza
    t = t.unsqueeze(1)  # [1,1,H,W]
    if device is not None:
        t = t.to(device)
    return t


def tensor_to_image(t: Tensor, is_mask: bool | None = None) -> Image.Image:
    """
    Converte un tensore in PIL.Image.
    - Accetta tensori [C,H,W], [B,C,H,W] con valori in [0,1].
    - Se is_mask è None, viene considerata maschera se C==1.
    - Per batch concatena le immagini lungo la larghezza.
    - Restituisce 'L' per maschere (C==1) e 'RGB' per immagini (C==3).
    """
    t = t.detach().clamp(0, 1)

    if t.ndim == 3:
        t = t.unsqueeze(0)  # [1,C,H,W]
    if t.ndim != 4:
        raise ValueError(f"tensor_to_image: tensore con {t.ndim} dimensioni non supportato")

    b, c, h, w = t.shape
    if is_mask is None:
        is_mask = (c == 1)

    t_cpu = (t * 255).to(dtype=torch.uint8, device="cpu")
    # [B,C,H,W] -> [H, B*W, C]
    arr = rearrange(t_cpu, "b c h w -> h (b w) c").numpy()

    if is_mask:
        # arr shape: (H, B*W, 1) -> squeeze last dim -> (H, B*W)
        arr_gray = arr.squeeze(-1)
        return Image.fromarray(arr_gray, mode="L")
    else:
        # arr shape: (H, B*W, 3)
        return Image.fromarray(arr)

#endregion