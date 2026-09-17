import time
import hydra


from src.engine import Engine
from omegaconf import DictConfig
# Assicurati di avere i tuoi import soliti (es. from src.engine import Engine)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    engine = Engine(cfg, is_train=True)
    
    # --- INIZIO TEST VELOCITÀ DATALOADER ---
    print("\n" + "="*40)
    print("TEST DATALOADER (NO GPU/TRAINING)")
    print("="*40)
    
    start_time = time.time()
    for i, batch in enumerate(engine.loader):
        elapsed = time.time() - start_time
        print(f"Batch {i} pronto in: {elapsed:.2f} secondi")
        
        if i == 5:
            print("Test concluso. Interrompo.")
            break
            
        start_time = time.time()
    # --- FINE TEST ---

    # engine.fit()  <-- COMMENTA QUESTO PER ORA


if __name__ == "__main__":
    main()