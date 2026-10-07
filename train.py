import hydra
from omegaconf import DictConfig

from src.engine_masked import Engine
import os

GPU = "1"

os.environ["EGL_DEVICE_ID"] = GPU  
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["HYDRA_FULL_ERROR"] = "1" # Debug
os.environ["PYOPENGL_PLATFORM"] = "egl"      # Rendering Option 


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    engine = Engine(cfg, is_train=True)
    engine.fit()


if __name__ == "__main__":
    main()