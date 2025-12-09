from pathlib import Path
from typing import List

from PIL.Image import Image

from tokenopt_generator.inpaiting_utils.inpainting import add_conf, ObjectiveType, Config,image_to_tensor,tensor_to_image,mask_to_tensor,ComposedCLIP,ReconstructionObjective
from tokenopt_generator.token_opt.tto.test_time_opt import TestTimeOptConfig,TestTimeOpt,CLIPObjective,MultiObjective

configs = [
    add_conf(
        name="C1_RESET",
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
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    ),
    add_conf(
        name="C3_RECON_STRONG_RESET",
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
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ReconstructionObjective,
                         ObjectiveType.ComposedCLIP]
    ),
    add_conf(
        name="C4_CLIPONLY_RESET",
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
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ComposedCLIP]
    ),
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
        enable_token_reset=True,
        reset_period=10,
        objective_types=[ObjectiveType.ComposedCLIP]
    )
]

def generate_inpainting(
    input_image_path: Path,
    mask_path: Path,
    prompt: str,
    num_generations: int,
    output_dir: Path,
):
    out_path_images=[]
    input_tns= image_to_tensor(input_image_path, device="cuda")
    mask_tns= mask_to_tensor(mask_path, device="cuda")
    input_masked=input_tns*mask_tns
    for name,config,objective_types in configs:
        objectives = []
        for obj_type, weight in zip(objective_types, config.objective_weights):
            if obj_type == ObjectiveType.ReconstructionObjective:
                recon_obj = ReconstructionObjective(input_masked, mask_tns)
                objectives.append(recon_obj)
            elif obj_type == ObjectiveType.ComposedCLIP:  # se e' un objective di ComposedCLIP e siamo in inpainting
                base_clip_obj = CLIPObjective(
                    prompt=prompt,
                    cfg_scale=config.cfg_scale,
                    num_augmentations=config.num_augmentations
                )  # creo la CLIPObjective di base
                orig_img = input_tns
                composed_clip_obj = ComposedCLIP(
                    base_clip_obj=base_clip_obj,
                    orig_img=orig_img,
                    mask_bin=mask_tns,
                    outside_grad=0.0
                )  # creo la ComposedCLIP
                objectives.append(composed_clip_obj)
            else:
                raise ValueError(f"Objective type {obj_type} not recognized")

        multi_objective = MultiObjective(objectives, config.objective_weights)
        tto = TestTimeOpt(config.tto_config, multi_objective)
        result_tns= tto(input_tns,mask_tns)
        result_img= tensor_to_image(result_tns)
        out_path = output_dir / f"{name}_result.png"
        result_img.save(out_path)
        out_path_images.append(out_path)
    return out_path_images




