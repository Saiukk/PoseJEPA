import time
import logging
import hydra

from src.engine import Engine
from omegaconf import DictConfig



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    engine = Engine(cfg, is_train=True)

    print("\n" + "="*40)
    print("TEST DATALOADER (NO GPU/TRAINING)")
    print("="*40)

    times = []
    start_time = time.time()
    for i, batch in enumerate(engine.loader):
        elapsed = time.time() - start_time
        if i >= 20:
            times.append(elapsed)
        print(f"Batch {i} pronto in: {elapsed:.2f} secondi")
        if i == 30:   # bastano pochi batch con DEBUG attivo, i log saranno tanti
            break
        start_time = time.time()

    import statistics
    print(f"\nMedia steady-state: {statistics.mean(times):.2f}s")
    print(f"Mediana: {statistics.median(times):.2f}s")

if __name__ == "__main__":
    main()