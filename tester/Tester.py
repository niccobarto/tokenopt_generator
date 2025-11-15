import enum
from pathlib import Path
import json
import torch
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


def _load_rgb_image_as_tensor(image_path: Path, size: int = 256, device = None) -> Tensor:
    img = Image.open(image_path).convert("RGB")
    t = (1.0 / 255.0) * torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1)
    # torchvision v2 expects size as sequence for some overloads; pass [size, size]
    t = tvf.resize(t, [size, size])
    t = tvf.center_crop(t, [size, size])
    t = t.unsqueeze(0)  # [1,3,H,W]
    if device is not None:
        t = t.to(device)
    return t


def image_to_tensor(image_path: Path, size: int = 256, device = None) -> Tensor:
    """
    Carica immagine RGB e restituisce tensore [1,3,H,W] float in [0,1].
    Per default (device=None) mantiene il tensore su CPU per risparmiare VRAM;
    passare device=DEVICE per spostarlo subito su GPU.
    """
    return _load_rgb_image_as_tensor(image_path, size=size, device=device)


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



class TTOTester:

    def __init__(self,tto):
        self.tto:TTOExecuter=tto
        self.is_inpaiting=self.tto.config.is_inpaiting

    def start_test(self,objects_to_test:list[str],images_types:list[str],dataset_path:Path,json_filename:str,output_path:Path):
        json_path=dataset_path/Path(json_filename) #es DATASET/inpaiting/testing1.json, informa il path del file json
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

                if self.is_inpaiting:  # se il test è di inpaiting
                    mask_tns = mask_to_tensor(Path(input_file_path) / test[
                        "mask_name"])  # carico la maschera del test in un tensore (deve essere con valori 0 e 1)

                    seed = orig_tns * mask_tns  # immagine mascherata (tensori in CPU)
                    # eseguo il test
                    test_results: list[(Tensor, float, str, Tensor)] = self._execute_test(seed, prompts,
                                                                                          mask_tns)
                    # salvo la maschera usata
                    mask = tensor_to_image(mask_tns,True)
                    mask.save(result_dir_path / Path(test["mask_name"]))  # salvo la maschera usata
                else:
                    # not inpainting
                    seed = orig_tns
                    # eseguo il test
                    test_results: list[(Tensor, float, str, Tensor)] = self._execute_test(seed,
                                                                                          prompts)
                out_tests = []
                prompt_number = 1
                # salvo i risultati del singolo test
                for result_img_tensor, loss, prompt, seed in test_results:

                    result_img = tensor_to_image(result_img_tensor)  # converto il tensore risultato in immagine
                    seed_img = tensor_to_image(
                        orig_tns * mask_tns if self.is_inpaiting else orig_tns)  # converto il tensore seed in immagine
                    create_side_by_side_with_caption(
                        # creo l'immagine affiancata con didascalia
                        left_img=seed_img,
                        right_img=result_img,
                        prompt=prompt,
                        value=loss,
                        out_path=result_dir_path / Path(f"prompt_{prompt_number}.png"),
                    )
                    out_single_test_json = {
                        "prompt_number": prompt_number,
                        "prompt": prompt,
                        "loss": loss,
                        "result_path": str(
                            result_dir_path / Path(f"prompt_{prompt_number}.png")
                        ),
                    }
                    out_tests.append(out_single_test_json)

                    prompt_number += 1

                out_tests_case.append(
                    {
                        "test_index": f"test_{test_index}",
                        "mask_path": str(result_dir_path / Path(test["mask_name"])) if self.is_inpaiting else None,
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
    def _execute_test(self, seed: Tensor, prompts: list[str], mask: Tensor=None) -> list[(Tensor, float, str, Tensor)]:
         results_combined=[]
         for prompt in prompts:
            # Impostiamo gli objective e lanciamo il TTO reale
            try:
                self.tto.set_objective(seed, prompt, mask)
                result, loss = self.tto.run(seed, mask)
            except Exception as e:
                # se fallisce l'esecuzione, includiamo comunque un placeholder nel risultato
                # e rialziamo l'eccezione dopo aver liberato risorse (la run già pulisce internamente)
                raise
            results_combined.append((result, loss, prompt, seed))
         return results_combined



"""
IMPORTANTE:
Le possibili combinazioni di objective per inpaiting sono:
- ReconstructionObjective | CLIPObjective
- ReconstructionObjective | ComposedCLIP
- ComposedCLIP
Quindi nei pesi passare sempre prima il peso di reconstruction e poi quello di CLIP/ComposedCLIP, altrimenti
solo quello di CLIP/ComposedCLIP.

Per not_inpaiting:
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
    is_inpaiting:bool
    seed_original:bool=False #se True si passa l'immagine originale al tto, altrimenti immagine mascherata
    objective_seed_original:bool=False #se True si passa l'immagine originale al CLIPObjective, altrimenti immagine mascherata

class ObjectiveType(enum.Enum):
    ReconstructionObjective=1
    CLIPObjective=2
    ComposedCLIP=3

class TTOExecuter:

    def __init__(self,config:Config,objectives_type:list[ObjectiveType],device):
        self.config:Config=config
        self.device=device
        self.objective: MultiObjective | None=None
        self.objectives_type=objectives_type
        if len(self.config.objective_weights)!=len(self.objectives_type): #controllo che il numero di pesi sia uguale al numero di objective
            raise ValueError("The number of objective weights must be equal to the number of objectives")

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
        objectives_list=[]
        for i,objective_type in enumerate(self.objectives_type): #scorro i tipi di objective che sono stati scelti
            if objective_type==ObjectiveType.ReconstructionObjective and self.config.is_inpaiting: #se e' un objective di ricostruzione e siamo in inpaiting
                if mask is None: #in questo caso DEVE esserci la maschera
                    raise ValueError("ReconstructionObjective requires a mask")
                masked_img=seed*mask #immagine mascherata
                recon_obj=ReconstructionObjective(masked_img,mask) #creo la ReconstructionObjective
                objectives_list.append(recon_obj) #la aggiungo alla lista
            elif objective_type==ObjectiveType.CLIPObjective: #se e' un objective di CLIP base
                clip_obj=CLIPObjective(
                    prompt=prompt,
                    neg_prompt="",
                    cfg_scale=self.config.cfg_scale,
                    num_augmentations=self.config.num_augmentations
                ) #creo la CLIPObjective
                objectives_list.append(clip_obj)
            elif objective_type==ObjectiveType.ComposedCLIP and self.config.is_inpaiting: #se e' un objective di ComposedCLIP e siamo in inpaiting
                base_clip_obj=CLIPObjective(
                    prompt=prompt,
                    neg_prompt="",
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
            else:
                raise ValueError(f"Objective type {objective_type} not recognized")
        if len(objectives_list)!=len(self.config.objective_weights): #controllo che il numero di pesi sia uguale al numero di objective creati
            raise ValueError("The number of objective weights must be equal to the number of objectives")

        """TENERE CONTO DELL'ORDINE DEI PESI RISPETTO AGLI OBJECTIVE"""
        self.objective=MultiObjective(objectives_list, self.config.objective_weights) #creo il MultiObjective con la lista di objective e i pesi corrispondenti


    def run(self,seed:Tensor,mask:Tensor=None)->(Tensor,float):
        if self.objective is None: #controllo che l'objective sia stato settato
            raise ValueError("Objectives not set. Call set_objective() before run().")
        if self.config.is_inpaiting: #se siamo in inpaiting
            if mask is None: #DEVE esserci una maschera in caso di inpaiting
                raise ValueError("Mask is required for inpainting")
            tto_input= seed if self.config.seed_original else seed*mask #immagine originale o mascherata a seconda del flag
        else:  # not in inpaiting
            tto_input=seed #immagine originale

        # Per risparmiare VRAM fino al momento dell'esecuzione spostiamo il tensore su device solo ora
        if mask is not None:
            mask = mask.to(self.device)

        tto = TestTimeOpt(
            config=self.config.tto_config,
            objective=self.objective
        ).to(self.device) # creo il TestTimeOpt con la config e l'objective

        # Eseguiamo il TTO proteggendo la pulizia: in caso di eccezione ripuliamo e rilanciamo
        try:
            result_img, loss = tto(tto_input)
        except Exception:
            try:
                del tto
            except Exception:
                pass
            # pulizia esplicita degli objectivi e della cache
            self.clean_after_test()
            hard_clear_cuda()
            raise
        else:
            # successo: puliamo e restituiamo
            try:
                del tto
            except Exception:
                pass
            self.clean_after_test()
            hard_clear_cuda()
            return result_img.to("cpu"), loss

    def clean_after_test(self):
        try:
            del self.objective #elimino l'objective per liberare memoria
        except Exception as e:
            self.objective=None
        self.objective=None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_json_configuration(self):
        config_dict={
            "tto_config":self.config.tto_config.__dict__,
            "objective_weights":self.config.objective_weights,
            "cfg_scale":self.config.cfg_scale,
            "num_augmentations":self.config.num_augmentations,
            "is_inpaiting":self.config.is_inpaiting,
            "seed_original":self.config.seed_original,
            "objective_seed_original":self.config.objective_seed_original,
            "objectives_type":[obj_type.name for obj_type in self.objectives_type]
        }
        return config_dict

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
#endregion


#----------Example of usage----------

# Global device: prefer CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


is_inpaiting=True
img_types=["clean"] # Options: "clean","real"
objects=["cup","table","vase","lamp"] # Options: fill with the objects you want to test

json_file="testing1.json" #DEVE STARE DENTRO dataset_path
dataset_pth=Path(f"../DATASET/{'inpaiting' if is_inpaiting else 'not_inpaiting'}")
out_pth=Path(f"../output/{'inpaiting' if is_inpaiting else 'not_inpaiting'}")


tto_config=TestTimeOptConfig(
    num_iter=351,
    ema_decay=0.98,
    lr=1e-1,
    enable_amp=True,
    reg_type="seed",
    reg_weight=0.05,
    token_noise=2e-4,
    vae_deterministic_sampling=True
    )
conf1=Config(
    tto_config=tto_config,
    objective_weights=[0.5,0.5],
    cfg_scale=1.2,
    num_augmentations=8,
    is_inpaiting=is_inpaiting,
    seed_original=False,
    objective_seed_original=False

)
objective_types=[ObjectiveType.ReconstructionObjective,ObjectiveType.CLIPObjective]

tto_ex=TTOExecuter(
    config=conf1,
    objectives_type=objective_types,
    device=DEVICE
)

tester=TTOTester(
    tto=tto_ex, #inserire qui l'oggetto TTOExecuter opportuno
)
tester.start_test(
    objects_to_test=objects,
    images_types=img_types,
    dataset_path=dataset_pth,
    json_filename=json_file,
    output_path=out_pth
)

