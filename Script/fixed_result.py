import pandas as pd

in_path = "/experiments/riccardop02/Inference/energy_lmo-test.csv"
out_path = "/experiments/riccardop02/Inference/energyfixed_lmo-test.csv"

df = pd.read_csv(in_path)

# Opzione A: tempo non misurato
df["time"] = -1

# Opzione B: tempo totale per immagine, uguale su tutte le righe
# df["time"] = df.groupby(["scene_id", "im_id"])["time"].transform("sum")

df.to_csv(out_path, index=False)