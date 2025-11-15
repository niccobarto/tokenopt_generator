import json
import sys
from pathlib import Path

INTRO_MARKER = "Inpainting con Token-Opt: guida e note"
INTRO_MD = f"""
# {INTRO_MARKER}

Questo notebook mostra come eseguire l'inpainting guidato da un obiettivo percettivo (es. CLIP/SigLIP) ottimizzando i token latenti di un tokenizer (TiTok/VQGAN) a test time.

Punti chiave:
- Fornisci un'immagine seed (es. 256x256) e una maschera binaria della stessa dimensione.
- La maschera seleziona la regione da riscrivere o da preservare: se i risultati sembrano invertiti, scambia 0/1 nella maschera.
- L'obiettivo percettivo guida la ricostruzione nella regione mascherata; usa augmentations per maggiore robustezza.
- EMA (smoothing) e AMP (mixed precision) possono stabilizzare e velocizzare l'ottimizzazione.

Suggerimenti pratici:
- Mantieni `num_iter` moderato (200-600) e regola `lr` per evitare instabilità.
- Se noti perdita di dettagli non mascherati, aumenta `ema_decay` o rafforza l'ancoraggio con `reg_type="seed"`.
- Assicurati che immagine, maschera e normalizzazioni siano coerenti con l'obiettivo selezionato.

""".strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python insert_md_cell.py <path_to_ipynb>")
        sys.exit(1)

    ipynb_path = Path(sys.argv[1])
    if not ipynb_path.exists():
        print(f"File not found: {ipynb_path}")
        sys.exit(2)

    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells")
    if cells is None:
        # Tentativo per vecchi formati (non comune)
        ws = nb.get("worksheets")
        if ws and isinstance(ws, list) and ws[0].get("cells"):
            cells = ws[0]["cells"]
            is_legacy = True
        else:
            print("Unable to locate cells array in notebook JSON")
            sys.exit(3)
    else:
        is_legacy = False

    # Verifica se la cella introduttiva è già presente
    def has_intro(cs):
        for c in cs:
            if c.get("cell_type") == "markdown":
                src = c.get("source")
                if isinstance(src, list):
                    text = "".join(src)
                else:
                    text = src or ""
                if INTRO_MARKER in text:
                    return True
        return False

    if has_intro(cells):
        print("Intro already present; no changes made.")
        return

    intro_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": INTRO_MD.splitlines(True),
    }

    # Inserisci in cima
    new_cells = [intro_cell] + cells

    if is_legacy:
        nb["worksheets"][0]["cells"] = new_cells
    else:
        nb["cells"] = new_cells

    with ipynb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Intro inserted at top of {ipynb_path}")


if __name__ == "__main__":
    main()

