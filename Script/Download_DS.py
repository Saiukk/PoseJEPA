import os
from huggingface_hub import snapshot_download

# Abilita il trasferimento ultra-veloce
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("download of T-LESS ...")

snapshot_download(
    repo_id="bop-benchmark/tless",
    repo_type="dataset",
    local_dir="/Experiments/riccardop02/datasets/tless",
    
    allow_patterns=[
            "*base.zip", 
            "*models*.zip", 
            "*test*.zip"
        ]
)

print("Download Done!")