"""
One-off: convert an old funk_svd.pkl (which stored live Surprise objects) into
the portable plain-numpy format the app now expects. Run once locally:

    venv/bin/python scripts/convert_svd.py

No retraining — it just re-serialises the already-learned parameters.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from config import MODELS_DIR

PATH = f"{MODELS_DIR}/funk_svd.pkl"

# Unpickle the old instance. Surprise must be importable for this step (it is,
# locally). We read .algo / .trainset off the reconstructed instance directly.
with open(PATH, "rb") as f:
    old = pickle.load(f)

if not hasattr(old, "algo"):
    print("funk_svd.pkl is already in the new format — nothing to do.")
    sys.exit(0)

algo     = old.algo
trainset = old.trainset

data = {
    "global_mean":  float(trainset.global_mean),
    "pu": np.asarray(algo.pu, dtype=np.float32),
    "bu": np.asarray(algo.bu, dtype=np.float32),
    "bi": np.asarray(algo.bi, dtype=np.float32),
    "qi": np.asarray(algo.qi, dtype=np.float32),
    "user_inner":   {trainset.to_raw_uid(u): u for u in range(trainset.n_users)},
    "inner_to_mid": {i: trainset.to_raw_iid(i) for i in range(trainset.n_items)},
}
data["mid_to_inner"] = {mid: i for i, mid in data["inner_to_mid"].items()}
data["user_seen"] = {
    trainset.to_raw_uid(u): {iid for iid, _ in trainset.ur[u]}
    for u in range(trainset.n_users)
}

with open(PATH, "wb") as f:
    pickle.dump(data, f)

print(f"Converted {PATH} to portable plain-numpy format.")
print(f"  users={len(data['user_inner'])}  items={len(data['inner_to_mid'])}  "
      f"factors={data['qi'].shape[1]}")
