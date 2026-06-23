import os
from huggingface_hub import snapshot_download


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

NAME = "hb"

print(f"download of {NAME} ...")

snapshot_download(
    repo_id=f"bop-benchmark/{NAME}",
    repo_type="dataset",
    local_dir=f"/Experiments/riccardop02/datasets/{NAME}",
    
    allow_patterns=[
            # "*base.zip", 
            # "*models*.zip", 
            # "*test*.zip"
            "hb_val_pri*"
        ]
)

print("Download Done!")