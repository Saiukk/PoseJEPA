from pathlib import Path

# Dataset
DATASET = 'icbin'
DATASET_ROOT = Path(f'/data/riccardop02/datasets/{DATASET}')
SPLIT = 'test'

# BOP Submission
PREDICTIONS_CSV = Path('/experiments/riccardop02/Inference/2026-11-11_14-33-15_icbin_energy_refSTEP3_GT/energy_icbin-test.csv')

EVAL_DIR = PREDICTIONS_CSV.with_suffix("")

# Dove salvare le analisi; non modifica i risultati esistenti.
OUTPUT_DIR = Path(f'/data/riccardop02/Qualitative/qualitative_{DATASET}/')

# Quanti esempi selezionare per categoria (massimo; se non ce ne sono abbastanza, meno).
N_PER_GROUP = 50

# Soglie SOLO per la selezione qualitativa, non cambiano i punteggi BOP ufficiali.
GOOD = 0.80
BAD = 0.20

# Le liste sono facili da modificare, ad esempio lascia solo ["overall_best", "overall_worst"].
GROUPS = [
    'overall_best', 'overall_worst',
    'vsd_best', 'vsd_worst',
    'mssd_best', 'mssd_worst',
    'mspd_best', 'mspd_worst',
    'mspd_good_mssd_bad', 'vsd_good_mssd_bad',
]

# Visualizzazione: wireframe campionato (non un rendering occlusion-aware).
MAX_TRIANGLES = 800
CROP_MARGIN = 0.20
