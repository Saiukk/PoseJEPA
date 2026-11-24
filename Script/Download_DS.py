import os
from huggingface_hub import snapshot_download


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

NAME = "itodd"

print(f"download of {NAME} ...")

snapshot_download(
    repo_id=f"bop-benchmark/{NAME}",
    repo_type="dataset",
    local_dir=f"/data/riccardop02/datasets/{NAME}",
    
    allow_patterns=[
            # "*base.zip", 
            # "*models*.zip", 
            # "*test*.zip"
            "*val*.zip"
           
        ]
)

print("Download Done!")