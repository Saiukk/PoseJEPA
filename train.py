import hydra
from omegaconf import DictConfig

from src.engine import Engine
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HYDRA_FULL_ERROR"] = "1" # Debug


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    engine = Engine(cfg, is_train=True)
    engine.fit()


if __name__ == "__main__":
    main()