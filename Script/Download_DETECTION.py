import os
import shutil
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = "bop-benchmark/bop_extra" 
FILE_PATH = "default_detections/classic_bop23_model_based_unseen/cnos-fastsam/cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json"
DESTINATION_DIR = "/Experiments/riccardop02/detection/cnos/"

print(f"Download di {FILE_PATH} dal repo {REPO_ID} ...")

try:
    cached_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILE_PATH,
        repo_type="dataset"
    )
    
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    only_file_name = os.path.basename(FILE_PATH)
    final_path = os.path.join(DESTINATION_DIR, only_file_name)
    
    shutil.copy2(cached_file, final_path)
    
    print(f"\nDownload completato! Nessuna cartella extra creata.")
    print(f"Il file si trova ESATTAMENTE qui: {final_path}")

except Exception as e:
    print(f"\nErrore durante il download: {e}")