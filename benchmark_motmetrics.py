import time

import motmetrics as mm

gt_dir = "/Users/mikel.brostrom/boxmot/data/benchmarks/MOT17/ablation"
pred_dir = "/Users/mikel.brostrom/boxmot/runs/mot/mot17/yolox_x_MOT17_ablation_lmbn_n_duke_boosttrack"
n_jobs = 8

start = time.perf_counter()

summary = mm.evaluate_motchallenge(
    gt_dir,
    pred_dir,
    n_jobs=n_jobs,
    progress=False,
)

elapsed = time.perf_counter() - start

print(summary["OVERALL"])
print(len(summary["OVERALL"]))
print(f"\nEvaluated in {elapsed:.3f}s using {n_jobs} workers")
