from typing import Dict, Any

from tester.Tester import (TTOTester,
                           TTOExecuter,
                           Config,
                           )
from inpaiting_utils.inpainting import ObjectiveType
import enum
from pathlib import Path
import json
import torch
from sympy import Tuple
from torch import Tensor,nn
import numpy as np
import torchvision.transforms.v2.functional as tvf
from PIL import Image
import gc
from tester.ImageSaver import create_side_by_side_with_caption
from einops import rearrange
from dataclasses import dataclass
from token_opt.tto.test_time_opt import (
                                        TestTimeOpt,
                                        TestTimeOptConfig,
                                        CLIPObjective,
                                        MultiObjective
                                    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_inpainting=True
configurations = []

# ============================================================
# Helper per aggiungere configurazioni
# ============================================================
def add_conf(name, tto_params, cfg_scale, num_aug, weights,
            seed_original, objective_seed_original,
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
        is_inpainting=is_inpainting,
        seed_original=seed_original,
        objective_seed_original=objective_seed_original,
        enable_token_reset=enable_token_reset,
        reset_period=reset_period,
    )

    configurations.append((name, conf, objective_types))

# ============================================================
img_types = ["clean"]  # Options: "clean","real"
objects = ["cup", "table", "vase", "lamp", "guitar", "plate", "wardrobe",
               "painting"]  # Options: fill with the objects you want to test
json_file = "dataset_inpainting.json"  # DEVE STARE DENTRO dataset_path


def main() -> None:

    # region Configurazioni
    # ============================================================
    # CONFIG C1 — baseline
    # ============================================================

    add_conf(
        name="C1_BASELINE_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.5e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1.0, 1.0],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=None,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # Reset leggero (periodo 10)
    # ============================================================

    add_conf(
        name="C1_BASELINE_RESET10",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.5e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1.0, 1.0],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # CONFIG C2 - Clip strong
    # ============================================================

    add_conf(
        name="C2_CLIPSTRONG_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=2,
        num_aug=8,
        weights=[1.0, 1.3],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # Reset leggero (periodo 10)
    # ============================================================

    add_conf(
        name="C2_CLIPSTRONG_RESET10",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=2,
        num_aug=8,
        weights=[1.0, 1.3],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # CONFIG C3 — Reconstruction più pesante
    # ============================================================
    add_conf(
        name="C3_RECON_STRONG_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.95,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.2,
        num_aug=8,
        weights=[1.5, 0.8],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # Reset leggero (periodo 10)
    # ============================================================
    add_conf(
        name="C3_RECON_STRONG_RESET10",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.95,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.2,
        num_aug=8,
        weights=[1.5, 0.8],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # CONFIG C4 — CLIPOnly
    # ============================================================
    add_conf(
        name="C4_CLIPONLY_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.95,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=10,
        objective_types=[ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # Reset leggero (periodo 10)
    # ============================================================

    add_conf(
        name="C4_CLIPONLY_RESET10",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.95,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # CONFIG C5 — LR più bassa (stabilità)
    # ============================================================
    add_conf(
        name="C5_LR_SMALL_NORESET",
        tto_params=dict(
            num_iter=501,
            ema_decay=0.97,
            lr=5e-2,  # lr più bassa
            enable_amp=True,
            reg_weight=2.5e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1.0, 1.0],  # [Reconstruction, ComposedCLIP]
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=None,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # Reset leggero (periodo 10)
    add_conf(
        name="C5_LR_SMALL_RESET10",
        tto_params=dict(
            num_iter=501,
            ema_decay=0.97,
            lr=5e-2,
            enable_amp=True,
            reg_weight=2.5e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1.0, 1.0],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # CONFIG C6 — EMA disattivato
    # ============================================================
    add_conf(
        name="C6_EMA_OFF_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.0,  # EMA off
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.5e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1.0, 1.0],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=None,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # Reset leggero (periodo 10)
    add_conf(
        name="C6_EMA_OFF_RESET10",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.0,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.5e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[1.0, 1.0],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # CONFIG C7 — Reconstruction light
    # ============================================================
    add_conf(
        name="C7_RECON_LIGHT_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[0.5, 1.0],  # recon più debole, CLIP più importante
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=None,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # Reset leggero (periodo 10)
    add_conf(
        name="C7_RECON_LIGHT_RESET10",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",
        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[0.5, 1.0],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )
    # ============================================================
    # CONFIG C8 — Prova con reg_type="zero"
    # ============================================================
    add_conf(
        name="C8_RECON_LIGHT_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.97,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="zero",
        ),
        cfg_scale=1.5,
        num_aug=8,
        weights=[0.5, 1.0],  # recon più debole, CLIP più importante
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=None,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # CONFIG C9 — CLIPOnly
    # ============================================================
    add_conf(
        name="C9CLIPONLYSTRONG_NORESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.98,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=3,
        num_aug=10,
        weights=[1],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=False,
        reset_period=10,
        objective_types=[ObjectiveType.ComposedCLIP]
    )

    # ============================================================
    # Reset leggero (periodo 10)
    # ============================================================

    add_conf(
        name="C9CLIPONLYSTRONG_RESET",
        tto_params=dict(
            num_iter=351,
            ema_decay=0.98,
            lr=1e-1,
            enable_amp=True,
            reg_weight=2.0e-2,
            token_noise=2e-4,
            reg_type="seed",

        ),
        cfg_scale=3,
        num_aug=10,
        weights=[1],
        seed_original=False,
        objective_seed_original=True,
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ComposedCLIP]
    )

    # endregion

    print("Created configurations:", [name for name, _, _ in configurations])

    dataset_pth = Path(f"DATASET/{'inpainting' if is_inpainting else 'not_inpainting'}")
    base_path = ask_base_path()
    data_root = base_path / dataset_pth
    results_root = base_path / "outputs" / f"{'inpainting' if is_inpainting else 'not_inpainting'}"

    print("\n=== Riepilogo path ===")
    print(f"Base path: {base_path}")
    print(f"Dataset:   {data_root}")
    print(f"Results:   {results_root}")
    print("======================\n")

    for cfg in configurations:
        run_configuration(cfg, data_root=data_root, results_root=results_root)



def run_configuration(cfg: (str, Config, list[ObjectiveType]), data_root: Path, results_root: Path) -> None:

    exp_name,conf,objective_types = cfg

    out_dir = results_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Running experiment: {exp_name}")
    print(f"    Data root:   {data_root}")
    print(f"    Results dir: {out_dir}")

    tto_ex = TTOExecuter(
        config=conf,
        objectives_type=objective_types,
        device=DEVICE
    )

    tester = TTOTester(
        tto=tto_ex,  # inserire qui l'oggetto TTOExecuter opportuno
    )
    tester.start_test(
        objects_to_test=objects,
        images_types=img_types,
        dataset_path=data_root,
        json_filename=json_file,
        output_path=out_dir
    )

def ask_base_path() -> Path:
    """
    Chiede all'utente, da terminale, la base del filepath.
    Su RunPod tipicamente sarà /workspace, che corrisponde al Network Volume.
    Se premi solo Invio, usa il default.
    """
    default_base = "/workspace"

    print("\n=== Configurazione base path ===")
    user_input = input(f"Base directory per dataset e risultati [{default_base}]: ").strip()

    if user_input == "":
        base_str = default_base
    else:
        base_str = user_input

    base_path = Path(base_str)

    if not base_path.exists():
        print(f"ATTENZIONE: il path {base_path} non esiste.")
        print("Controlla di essere dentro il Pod giusto e che il volume sia montato.")
        # Se vuoi essere cattivo, puoi fare:
        # import sys; sys.exit(1)

    return base_path

if __name__ == "__main__":
    main()