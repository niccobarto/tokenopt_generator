import enum
from pathlib import Path
import json
import random

import torch
from jaxtyping import Float
from torch import Tensor,nn
import numpy as np
import torchvision.transforms.v2.functional as tvf
from PIL import Image
import gc
from tokenopt_generator.tester.ImageSaver import create_side_by_side_with_caption
from einops import rearrange
from dataclasses import dataclass
from tokenopt_generator.token_opt.tto.test_time_opt import (
                                        TestTimeOpt,
                                        TestTimeOptConfig,
                                        CLIPObjective,
                                        SigLIPObjective)
from typing import cast, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from tokenopt_generator.inpaiting_utils.inpainting import (ReconstructionObjective,
                                                           ComposedCLIP,tensor_to_image,mask_to_tensor,image_to_tensor,ObjectiveType)

class LossPlotter:
    """
    Plotter pensato per un singolo test.
    - Creare una nuova istanza o chiamare `.reset()` quando si inizia un nuovo test.
    - add_values(values, weights=None): registra le loss di un prompt.
    - generate_plot(): ritorna due PIL.Image (raw, weighted).
    - save_plots(dir_path, prefix): salva le due immagini su disco.
    """
    def __init__(self, objective_names: list[str]):
        self.objective_names = list(objective_names)
        self.num_loss = len(self.objective_names)
        self.per_obj: list[list[float]] = [[] for _ in range(self.num_loss)]
        self.per_obj_weighted: list[list[float]] = [[] for _ in range(self.num_loss)]
        self.totals: list[float] = []
        self.totals_weighted: list[float] = []
        self.weights_history: list[list[float]] = []

    def add_values(self, values: list[float], weights: Optional[list[float]] = None):
        if len(values) != self.num_loss:
            raise ValueError("values length must match number of objectives")
        if weights is None:
            weights = [1.0] * self.num_loss
        if len(weights) != self.num_loss:
            raise ValueError("weights length must match number of objectives")

        for i, v in enumerate(values):
            self.per_obj[i].append(float(v))
            self.per_obj_weighted[i].append(float(v) * float(weights[i]))

        self.totals.append(float(sum(values)))
        self.totals_weighted.append(float(sum(v * w for v, w in zip(values, weights))))
        self.weights_history.append([float(w) for w in weights])

    def reset(self):
        """Svuota lo storico: chiamare all'inizio di ogni nuovo test."""
        self.per_obj = [[] for _ in range(self.num_loss)]
        self.per_obj_weighted = [[] for _ in range(self.num_loss)]
        self.totals = []
        self.totals_weighted = []
        self.weights_history = []

    def _fig_to_image(self, fig: Figure) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        buf.close()
        plt.close(fig)
        return img

    def generate_plot(self, figsize=(10, 4)) -> tuple[Image.Image, Image.Image]:
        """Genera e restituisce (img_raw, img_weighted) per l'attuale storico (singolo test).
        Nel plot pesato vengono mostrati anche (in tratteggio) i valori raw per confronto.
        """
        steps = list(range(1, len(self.totals) + 1))

        # Raw plot
        fig1, ax1 = plt.subplots(figsize=figsize)
        for i, name in enumerate(self.objective_names):
            ax1.plot(steps, self.per_obj[i], label=name, linewidth=1)
        if self.totals:
            ax1.plot(steps, self.totals, label="total (raw)", color="black", linewidth=2)
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Loss per objective (raw) + total")
        ax1.legend(loc="best", fontsize="small")
        img_raw = self._fig_to_image(fig1)

        # Weighted plot
        fig2, ax2 = plt.subplots(figsize=figsize)
        last_weights = self.weights_history[-1] if self.weights_history else [1.0] * self.num_loss
        for i, name in enumerate(self.objective_names):
            # curve pesata
            label_w = f"{name} (w={last_weights[i]:.2f})"
            ax2.plot(steps, self.per_obj_weighted[i], label=label_w, linewidth=1)
            # overlay raw per confronto visivo se ci sono valori
            ax2.plot(steps, self.per_obj[i], linestyle="--", alpha=0.3, linewidth=1)
        if self.totals_weighted:
            ax2.plot(steps, self.totals_weighted, label="total (weighted)", color="black", linewidth=2)
        ax2.set_xlabel("step")
        ax2.set_ylabel("loss (weighted)")
        ax2.set_title("Loss pesata per objective + total (raw tratteggiato)")

        # Mantieni lo stesso range y del plot raw per evidenziare la scala
        try:
            raw_values_all = [v for series in self.per_obj for v in series]
            if raw_values_all:
                y_min = min(raw_values_all)
                y_max = max(raw_values_all)
                if y_min != y_max:
                    ax2.set_ylim(y_min, y_max)
        except Exception:
            pass
        ax2.legend(loc="best", fontsize="small")
        img_weighted = self._fig_to_image(fig2)

        return img_raw, img_weighted

    def save_plots(self, out_raw_path, out_weighted_path):
        """Genera e salva i plot direttamente su disco."""
        img_raw, img_weighted = self.generate_plot()
        img_raw.save(out_raw_path)
        img_weighted.save(out_weighted_path)

# Rende l'istanza globale accessibile ovunque nel modulo
loss_plotter_global: LossPlotter | None = None

class MultiObjective(nn.Module):
    def __init__(self, objectives: list[nn.Module], weights: list[float]):
        super().__init__()
        self.weights = weights
        self.objectives = nn.ModuleList(objectives)

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        # Calcola le loss per ogni objective e registra i valori nel plotter globale
        losses_per_obj: list[Tensor] = []  # ciascuna [B]
        loss = torch.zeros_like(img[:, 0, 0, 0])
        for w, o in zip(self.weights, self.objectives):
            li = o(img)  # [B]
            losses_per_obj.append(li)
            loss = loss + w * li

        # Registra (una volta per forward) le loss nel plotter globale
        try:
            global loss_plotter_global
            if loss_plotter_global is not None:
                # convertiamo a float per ogni objective, aggregando sul batch (mean)
                values = [float(li.detach().mean().item()) for li in losses_per_obj]
                loss_plotter_global.add_values(values=values, weights=[float(w) for w in self.weights])
        except Exception:
            pass
        return loss

def hard_clear_cuda(names=()):
    """Elimina variabili globali indicate e forza la pulizia della memoria GPU/CPU.
    Passare i nomi delle variabili globali (stringhe) da eliminare se necessario.
    """
    import gc, torch
    g = globals()
    for n in list(names):
        if n in g:
            try:
                del g[n]
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

class TTOTester:

    def __init__(self,tto):
        self.tto:TTOExecuter=tto
        self.is_inpainting=self.tto.config.is_inpainting

    def start_test(self,objects_to_test:list[str],images_types:list[str],dataset_path:Path,json_filename:str,output_path:Path):
        json_path=dataset_path/Path(json_filename) #es DATASET/inpainting/dataset_inpainting.json, informa il path del file json
        json_output={"images_types":[],
                     "config": self.tto.get_json_configuration(),
                     } #dizionario che conterra' i risultati dei test
        file_json = json.load(open(json_path)) #carico il file json
        types=file_json["images_types"] #es [clean,real]
        for typ in types: #scorro i tipi di immagini
            if typ["images_type"] in images_types: #se il tipo di immagine e' tra quelli da testare
                img_type = typ["images_type"] #es clean/real
                out_dataset_json=[]
                dataset = typ["dataset"] #prendo il dataset associato al tipo di immagine
                for obj in dataset: #scorro gli oggetti del dataset
                    obj_name=obj["object"]
                    print(f"###--- Testing object: {obj_name} of type {img_type} ---###")
                    if  obj_name in objects_to_test:
                        output_path_json=self.test_single_object(obj,dataset_path,output_path)
                        out_obj_cases_json={
                            "object":obj_name,
                            "cases":output_path_json
                        }
                        out_dataset_json.append(out_obj_cases_json)
                json_output["images_types"].append({
                    "images_type": img_type,
                    "dataset": out_dataset_json
                })  # aggiungo il tipo di immagine al json di output

                # Scriviamo il json di output (usiamo json.dumps per evitare ambiguità sul tipo di file per il checker)
                with open(output_path / Path(f"results.json"), "w", encoding="utf-8") as outfile:
                    outfile.write(json.dumps(json_output, indent=4))

    def test_single_object(self,obj,dataset_path:Path,output_path:Path):
        cases = obj["cases"]  # prendo i casi associati all'oggetto che ha passato il controllo di flag di testing
        output_cases=[]
        for case in cases:  # scorro i casi che un determinato oggetto ha (casi=varianti dell'oggetto in immagini)
            case_name=case["case"]
            print(" #-- Testing case:",case_name," --#")
            info = case["info"]  # prendo le info associate al caso (
            files_path = info["files_path"]  # path relativo alla directory del caso, es: clean/cup/cup1
            input_file_path = Path(dataset_path) / Path(
                files_path)  # path relativo all'intero dataset per la directory di un caso
            output_file_path = Path(output_path) / Path(
                files_path)  # path relativo alla directory di output del caso

            orig_path = input_file_path / info["image_name"]  # path dell'immagine originale
            orig_image = Image.open(orig_path)  # apro l'immagine originale

            output_file_path.mkdir(parents=True, exist_ok=True)  # creo la directory di output per il caso
            orig_image.save(output_file_path / Path("original.png"))  # salvo l'immagine nella directory di output

            # NOTE: per risparmiare VRAM manteniamo per default orig e mask su CPU
            orig_tns = image_to_tensor(orig_path)  # carico l'immagine originale in un tensore (CPU di default)
            tests = case["tests"]  # prendo i test associati al caso
            test_index = 1

            out_tests_case=[]
            for test in tests:  # scorro i test associati al caso

                prompts = test["prompts"]  # prendo i prompt associati al test

                result_dir_path = output_file_path / Path(
                    f"test_{test_index}")  # ad esempio output/clean/cup/cup1/test_1
                result_dir_path.mkdir(parents=True,
                                      exist_ok=True)  # creo la directory di output per il test_i-esimo

                mask_tns = None

                if self.is_inpainting:  # se il test è di inpainting
                    mask_tns = mask_to_tensor(Path(input_file_path) / test[
                        "mask_name"])  # carico la maschera del test in un tensore (deve essere con valori 0 e 1)

                    seed = orig_tns * mask_tns  # immagine mascherata (tensori in CPU)
                    # eseguo il test
                    print(f"-- Executing inpainting test for obj: {obj['object']} and mask: {test['mask_name']} --")
                    test_results: list[tuple[Tensor, float, str, Tensor, Optional[tuple[Image.Image, Image.Image]]]] = self._execute_test(seed, prompts,
                                                                                          mask_tns)
                    # salvo la maschera usata
                    mask = tensor_to_image(mask_tns,True)
                    mask.save(result_dir_path / Path(test["mask_name"]))  # salvo la maschera usata
                else:
                    # not inpainting
                    seed = orig_tns
                    print(f"-- Executing not-inpainting test for obj: {obj['object']} --")
                    # eseguo il test
                    test_results: list[tuple[Tensor, float, str, Tensor, Optional[tuple[Image.Image, Image.Image]]]] = self._execute_test(seed,
                                                                                          prompts)
                out_tests = []
                prompt_number = 1
                # salvo i risultati del singolo test
                for result_img_tensor, clip_score, prompt, seed, plots in test_results:

                    result_img = tensor_to_image(result_img_tensor)  # converto il tensore risultato in immagine
                    seed_img = tensor_to_image(
                        orig_tns * mask_tns if self.is_inpainting else orig_tns)  # converto il tensore seed in immagine
                    create_side_by_side_with_caption(
                        # creo l'immagine affiancata con didascalia
                        left_img=seed_img,
                        right_img=result_img,
                        prompt=prompt,
                        value=clip_score,
                        out_path=result_dir_path / Path(f"prompt_{prompt_number}.png"),
                    )

                    # Salviamo anche l'immagine risultante "non-merged" in una sottocartella results_not_merged
                    not_merged_dir = result_dir_path / Path("results_not_merged")
                    not_merged_dir.mkdir(parents=True, exist_ok=True)
                    # Salviamo l'immagine singola con estensione .png
                    result_img.save(not_merged_dir / Path(f"prompt_{prompt_number}.png"))

                    if plots is not None:
                        raw_plot, weighted_plot = plots
                        # Salviamo i plots nella sottocartella 'plots' del test, con nomi richiesti
                        plots_dir = result_dir_path / Path("plots")
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        raw_plot.save(plots_dir / Path(f"prompt_{prompt_number}.png"))
                        weighted_plot.save(plots_dir / Path(f"prompt_{prompt_number}_weights.png"))
                    out_single_test_json = {
                        "prompt_number": prompt_number,
                        "prompt": prompt,
                        "clip_score": clip_score,
                        "result_path": str(
                            result_dir_path / Path(f"prompt_{prompt_number}.png")
                        ),
                    }
                    out_tests.append(out_single_test_json)

                    prompt_number += 1

                out_tests_case.append(
                    {
                        "test_index": f"test_{test_index}",
                        "mask_path": str(result_dir_path / Path(test["mask_name"])) if self.is_inpainting else None,
                        "tests": out_tests
                    })
                test_index += 1

            output_case = {
                "case": case_name,
                "info":{
                "case_path":str(output_file_path),
                "image_name":info["image_name"],
                },
                "tests":out_tests_case,
            }
            output_cases.append(output_case)
        return output_cases

    def _execute_test(self, seed: Tensor, prompts: list[str], mask: Tensor=None) -> list[tuple[Tensor, float, str, Tensor, Optional[tuple[Image.Image, Image.Image]]]]:
         results_combined=[]
         for prompt in prompts:
            # Impostiamo gli objective e lanciamo il TTO reale
            try:
                self.tto.set_objective(seed, prompt, mask)
                print("- Running TTO generation -")
                result_img = self.tto.run(seed, mask)
                print("- TTO generation completed -")
                if mask is not None:
                    # se la mask ha un solo canale, espandila sui 3 canali dell'immagine
                    if mask.shape[1] == 1 and result_img.shape[1] == 3:
                        mask_eval = mask.expand_as(result_img)
                    else:
                        mask_eval = mask

                    result_img = result_img * (1 - mask_eval) + seed * mask_eval  # ricomponiamo l'immagine finale con il seed nelle aree mascherate
                similarity=self.evaluate_image_similarity(result_img,prompt)
                # Genera i plot del singolo test (storico corrente) se il plotter è disponibile
                global loss_plotter_global
                plots: Optional[tuple[Image.Image, Image.Image]] = None
                if loss_plotter_global is not None:
                    plots = loss_plotter_global.generate_plot()
            except Exception as e:
                # se fallisce l'esecuzione, includiamo comunque un placeholder nel risultato
                # e rialziamo l'eccezione dopo aver liberato risorse (la run già pulisce internamente)
                raise
            results_combined.append((result_img, similarity, prompt, seed,plots))
         return results_combined

    def evaluate_image_similarity(self, img:Tensor,prompt:str)->float:
        # 3) costruisci il CLIPObjective per la valutazione
        # Use the executor's device (stored in the TTOExecuter instance)
        assert self.tto is not None, "executor (TTOExecuter) missing"
        image_evaluator = CLIPObjective(
            prompt=prompt,
            cfg_scale=1.0,
            num_augmentations=1
        ).to(self.tto.device).eval()

        with torch.no_grad():
            loss = image_evaluator(img.to(self.tto.device))  # shape [B]
            loss = loss.mean()  # scalare
            clip_score = (-loss).item()  # CLIPScore vero: similarity img_final vs prompt
        return clip_score


"""
IMPORTANTE:
Le possibili combinazioni di objective per inpainting sono:
- ReconstructionObjective | CLIPObjective
- ReconstructionObjective | ComposedCLIP
- ComposedCLIP
Quindi nei pesi passare sempre prima il peso di reconstruction e poi quello di CLIP/ComposedCLIP, altrimenti
solo quello di CLIP/ComposedCLIP.

Per not_inpainting:
- CLIPObjective


orig_seed==True -> si passa l'immagine originale al tto, altrimenti immagine mascherata
objective_seed==True -> si passa l'immagine originale al CLIPObjective, altrimenti immagine mascherata
"""
@dataclass
class Config:
    tto_config:TestTimeOptConfig #TestTimeOptConfig
    objective_weights:list[float] #lista di pesi peg li objectives
    cfg_scale:float
    num_augmentations:int
    is_inpainting:bool
    two_phase_generation:bool=False
    seed_original:bool=False #se True si passa l'immagine originale al tto, altrimenti immagine mascherata
    objective_seed_original:bool=False #se True si passa l'immagine originale al CLIPObjective, altrimenti immagine mascherata
    enable_token_reset:bool=True
    reset_period:int=5 if enable_token_reset else None,


class TTOExecuter:

    def __init__(self,config:Config,objectives_type:list[ObjectiveType],device):
        self.config:Config=config
        self.device=device
        self.objective: MultiObjective | None=None
        self.objectives_type=objectives_type
        # Persistent heavy model (TestTimeOpt) - keep it alive across tests to avoid repeated reloading
        # Annotate explicitly so static checkers know the expected type
        self.tto: TestTimeOpt | None = None
        if len(self.config.objective_weights)!=len(self.objectives_type): #controllo che il numero di pesi sia uguale al numero di objective
            raise ValueError("The number of objective weights must be equal to the number of objectives")
        names = []
        for obj in self.objectives_type:
            names.append(obj.name)

        # Inizializza davvero l'istanza globale del plotter con i nomi degli obiettivi
        global loss_plotter_global
        loss_plotter_global = LossPlotter(names)

        # Create a single TestTimeOpt instance and keep it for the lifetime of the executor.
        # We pass a dummy objective for initialization and will replace it in set_objective().
        # Constructing the heavy model here avoids reloading it on every run.
        # If construction fails, ensure the attribute exists as None so callers can detect it


    """
    ReconstructionObjective chiede: 
    -l'immagine mascherata 
    -la maschera come parametri

    CLIPObjective chiede:
    -prompt
    -neg_prompt
    -cfg_scale
    -num-augumentations


    
    ComposedCLIP chiede:
    -CLIPObjective (quindi tutti i suoi parametri)
    -l'immagine originale/mascherata
    -la maschera
    """
    def set_objective(self, seed: Tensor, prompt: str, mask: Tensor):
        # Reset del plotter a inizio test singolo, così accumula solo i valori del test corrente
        global loss_plotter_global
        if loss_plotter_global is not None:
            loss_plotter_global.reset()
        objectives_list=[]
        for objective_type in self.objectives_type:
            if objective_type==ObjectiveType.ReconstructionObjective and self.config.is_inpainting:
                if mask is None:
                    raise ValueError("ReconstructionObjective requires a mask")
                masked_img=seed*mask
                recon_obj=ReconstructionObjective(masked_img,mask)
                objectives_list.append(recon_obj)
            elif objective_type==ObjectiveType.CLIPObjective:
                clip_obj=CLIPObjective(
                    prompt=prompt,
                    cfg_scale=self.config.cfg_scale,
                    num_augmentations=self.config.num_augmentations
                ) #creo la CLIPObjective
                objectives_list.append(clip_obj)
            elif objective_type==ObjectiveType.ComposedCLIP and self.config.is_inpainting: #se e' un objective di ComposedCLIP e siamo in inpainting
                base_clip_obj=CLIPObjective(
                    prompt=prompt,
                    cfg_scale=self.config.cfg_scale,
                    num_augmentations=self.config.num_augmentations
                ) #creo la CLIPObjective di base
                if mask is None: #in questo caso DEVE esserci la maschera
                    raise ValueError("ComposedCLIP requires a mask")
                orig_img=seed if self.config.objective_seed_original else seed*mask #immagine originale o mascherata a seconda del flag
                composed_clip_obj=ComposedCLIP(
                    base_clip_obj=base_clip_obj,
                    orig_img=orig_img,
                    mask_bin=mask,
                    outside_grad=0.0
                ) #creo la ComposedCLIP
                objectives_list.append(composed_clip_obj)
            elif objective_type==ObjectiveType.ComposedSiglip:
                siglip_base=SigLIPObjective(
                    prompt=prompt,
                    num_augmentations=self.config.num_augmentations
                ) #creo la SigLIPObjective

                if mask is None:
                    raise ValueError("ComposedSiglip requires a mask")
                orig_img=seed if self.config.objective_seed_original else seed*mask
                siglip_obj=ComposedCLIP(
                    base_clip_obj=siglip_base,
                    orig_img=orig_img,
                    mask_bin=mask,
                    outside_grad=0.0
                ) #creo la ComposedCLIP con SigLIP come base
                objectives_list.append(siglip_obj)
            else:
                raise ValueError(f"Objective type {objective_type} not recognized")
        if len(objectives_list)!=len(self.config.objective_weights): #controllo che il numero di pesi sia uguale al numero di objective creati
            raise ValueError("The number of objective weights must be equal to the number of objectives")
        # Assemble the MultiObjective and try to move module objects to the target device.
        # Some objective-like objects (eg ReconstructionObjective) contain plain tensors and
        # may not support `.to()`; handle that gracefully.
        # Move Objective components to target device when possible, then create MultiObjective
        for i, o in enumerate(objectives_list):
            try:
                objectives_list[i] = o.to(self.device)
            except Exception:
                # leave as-is; those objects may handle device placement in forward()
                objectives_list[i] = o

        # Create the MultiObjective instance using (possibly moved) submodules
        self.objective = MultiObjective(objectives_list, self.config.objective_weights)
        # Ensure that the MultiObjective container itself is on the target device (best-effort)
        try:
            self.objective = self.objective.to(self.device)
        except Exception:
            # if `.to()` fails, it likely contains non-module objects; forward() will handle device placement
            pass

        """TENERE CONTO DELL'ORDINE DEI PESI RISPETTO AGLI OBJECTIVE"""
        self.objective=MultiObjective(objectives_list, self.config.objective_weights) #creo il MultiObjective con la lista di objective e i pesi corrispondenti

        # Lazy-create the heavy TestTimeOpt model the first time we have an objective.
        # Creation can fail (eg out-of-memory); surface the error but keep self.tto None
        # so callers can retry or handle the situation.
        if self.tto is None:
            try:
                # Create heavy TestTimeOpt once and move it to device
                self.tto = TestTimeOpt(
                    config=self.config.tto_config,
                    objective=self.objective,
                ).to(self.device)
            except Exception:
                # keep self.tto as None and re-raise so the caller sees the original error
                raise
        else:
            # Assign the freshly built objective to the persistent tto instance
            # Make sure the objective is placed on the same device as the tto before assignment
            try:
                self.objective = self.objective.to(self.device)
            except Exception:
                pass
            # Assign - this will register the module inside the persistent TestTimeOpt
            self.tto.objective = self.objective

            # Best-effort: ensure the persistent model and its new submodules are consistent on device
            try:
                self.tto.to(self.device)
            except Exception:
                pass

    def run(self,seed:Tensor,mask:Tensor=None)->Tensor:
        if self.objective is None:
            raise ValueError("Objectives not set. Call set_objective() before run().")
        if self.config.is_inpainting:
            if mask is None:
                raise ValueError("Mask is required for inpainting")
            # Spostiamo seed e mask sul device di esecuzione (GPU/CPU) qui, con assegnamento
            # in modo che tutte le operazioni element-wise successive usino lo stesso device.
            seed = seed.to(self.device)
            mask = mask.to(self.device)

            # Costruiamo qui l'input per il TTO (immagine originale o mascherata) una volta che
            # seed e mask sono sullo stesso device.
            tto_input = seed if self.config.seed_original else seed * mask
        else:
            tto_input = seed.to(self.device)

        token_reset = None
        if self.config.is_inpainting:
            if self.config.enable_token_reset:
                # Creiamo la masked image per il TokenResetter sullo stesso device
                masked_for_reset = seed * mask
                # assert to satisfy type-checkers: self.tto must exist here
                assert self.tto is not None, "persistent TestTimeOpt instance missing"
                token_reset = TokenResetter(
                    titok=cast(TestTimeOpt, self.tto).titok,
                    masked_img=masked_for_reset,
                    mask=mask,
                    reset_period=self.config.reset_period
                )
        # Sanity check: persistent tto must exist at this point
        if self.tto is None:
            raise RuntimeError("Internal error: persistent TestTimeOpt instance is missing. Ensure set_objective was called successfully.")

        try:
            # Call the persistent TestTimeOpt instance
            assert self.tto is not None

            # seed PRIMA di ogni test
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            random.seed(0)
            np.random.seed(0)

            if not self.config.two_phase_generation:
                result_img = cast(TestTimeOpt, self.tto)(
                    seed=tto_input,
                    token_reset_callback=token_reset)
            else:#GENERAZIONE IN DUE FASI
                clip_weight=self.config.objective_weights[1]
                self.objective.weights[1]=0.0 #disabilito temporaneamente il peso di CLIPObjective
                steps=self.config.tto_config.num_iter
                self.tto.config.num_iter=201 #prima fase 200 step
                result_img = cast(TestTimeOpt, self.tto)(
                    seed=tto_input,
                    token_reset_callback=token_reset)

                self.objective.weights[1]=clip_weight #riabilito il peso di CLIP
                self.tto.config.num_iter=steps #ripristino il numero di step originale
                result_img = cast(TestTimeOpt, self.tto)(
                    seed=result_img,
                    token_reset_callback=token_reset)

        except Exception:
            # do not delete the persistent tto; just free transient resources
            self.clean_after_test()
            hard_clear_cuda()
            raise
        else:
            # keep the persistent tto alive for reuse; only free transient resources
            self.clean_after_test()
            hard_clear_cuda()
            return result_img.to("cpu")

    def clean_after_test(self):
        """Liberiamo le risorse temporanee create durante il singolo test.

        Nota: non cancelliamo `self.tto` (il modello pesante persistente) — lo teniamo vivo
        per evitare costose ricostruzioni. Viene rimosso solo l'Objective temporaneo e forzata
        la pulizia della cache CUDA.
        """
        try:
            if hasattr(self, "objective"):
                # rimuoviamo il riferimento all'oggetto objective per permettere il GC
                del self.objective
        except Exception:
            pass
        self.objective = None
        # Rimuoviamo il riferimento all'objective anche dentro l'istanza persistente TestTimeOpt
        try:
            if self.tto is not None and hasattr(self.tto, "objective"):
                try:
                    delattr(self.tto, "objective")
                except Exception:
                    try:
                        del self.tto.objective
                    except Exception:
                        pass
        except Exception:
            pass
        # Forziamo il GC locale
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Metodi di utilità per debugging
    def debug_state(self):
        """Ritorna informazioni utili per il debug (tipo e device del tto)."""
        info = {
            "has_tto": self.tto is not None,
            "tto_type": type(self.tto).__name__ if self.tto is not None else None,
            "tto_device": "",
            "objective_set": self.objective is not None,
        }
        if self.tto is not None:
            try:
                # proviamo a ottenere device dal primo parametro del modello
                for p in self.tto.parameters():
                    info["tto_device"] = str(p.device)
                    break
            except Exception:
                info["tto_device"] = None
        return info

    def destroy_tto(self):
        """Elimina l'istanza pesante di TestTimeOpt per liberare memoria GPU/CPU.

        Usare quando si è sicuri di non dover più riutilizzare il modello (es. alla fine di una
        sessione di test). Dopo questa chiamata, il TTO verrà ricreato al prossimo
        `set_objective()` se necessario.
        """
        try:
            if self.tto is not None:
                # rimuoviamo riferimenti al modello e forziamo GC
                del self.tto
        except Exception:
            pass
        self.tto = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_json_configuration(self):
        config_dict={
            "tto_config":self.config.tto_config.__dict__,
            "objective_weights":self.config.objective_weights,
            "cfg_scale":self.config.cfg_scale,
            "num_augmentations":self.config.num_augmentations,
            "is_inpainting":self.config.is_inpainting,
            "enable_token_reset":self.config.enable_token_reset,
            "reset_period":self.config.reset_period,
            "seed_original":self.config.seed_original,
            "objective_seed_original":self.config.objective_seed_original,
            "objectives_type":[obj_type.name for obj_type in self.objectives_type]
        }
        return config_dict


class TokenResetter:
    def __init__(self, titok, masked_img, mask, reset_period=5):
        self.titok = titok
        self.masked_img = masked_img
        self.mask = mask
        self.reset_period = reset_period

    @torch.no_grad()
    def __call__(self, info):
        if info.i % self.reset_period != 0:
            return
        dec_reset = (1. - self.mask) * info.img + self.masked_img
        return self.titok.encoder(dec_reset, self.titok.latent_tokens)
