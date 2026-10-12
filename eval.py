import os, sys, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["BOP_PATH"] = "/data/riccardop02/datasets"
os.environ["BOP_RESULTS_PATH"] = "/experiments/riccardop02/Inference"
os.environ["BOP_EVAL_PATH"] = "/experiments/riccardop02/Inference"
os.environ["BOP_OUTPUT_PATH"] = "/experiments/riccardop02/Inference"

subprocess.run(
    [sys.executable, "scripts/eval_bop19_pose.py",
     "--renderer_type", "vispy",
     "--result_filenames", "energyfixed_lmo-test.csv"],
    cwd="/home/riccardop02/JEPA/bop_toolkit",
    check=True,
)