"""
Path module — va in: /Experiments/riccardop02/JEPA/Essential/path.py

Unico posto che conosce la geografia del disco. Tutto il resto del codice
importa da qui e non scrive mai path assoluti.
"""
from pathlib import Path

# --- radici (assolute, sul server) ---
DATA    = Path("/Experiments/riccardop02/datasets")          # icbin, xyzibd, ... (sola lettura)
OUTPUTS = Path("/Experiments/riccardop02/JEPA/Renders")      # tutto rigenerabile
RENDERS = OUTPUTS / "renders"

# crea le cartelle di output se non esistono
RENDERS.mkdir(parents=True, exist_ok=True)


def models_dir(name: str) -> Path:
    """Cartella delle mesh di un dataset BOP: .../<name>/models/obj_000001.ply ..."""
    return DATA / name / "models"


def scenes_dir(name: str, split: str = "test") -> Path:
    """Cartella delle scene reali (serve dopo, per il sim-to-real)."""
    return DATA / name / split