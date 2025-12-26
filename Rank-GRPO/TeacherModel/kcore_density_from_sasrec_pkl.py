import argparse
import pickle
from collections import Counter
from tqdm import tqdm

def compute_density(data, n_items):
    U = len(data)
    interactions = sum(len(x["sequence"]) for x in data)
    I = n_items
    dens = interactions / (U * I) if U > 0 and I > 0 else 0.0
    return U, I, interactions, dens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True)
    ap.add_argument("--k", type=int, default=10, help="k-core threshold")
    ap.add_argument("--max_iter", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        pkg = pickle.load(f)

    raw = pkg["data"]
    # n_items in your pkg is item id space size; but some ids may disappear after filtering
    # We'll recompute actual unique item count after k-core.
    print(f"Loaded users={len(raw)}")

    # Build user->sequence list (already)
    data = [{"user_id": str(x["user_id"]), "sequence": [int(t) for t in x["sequence"] if int(t) != 0]} for x in raw]
    # Drop too short
    data = [x for x in data if len(x["sequence"]) > 0]

    print("Initial density on full data:")
    # unique item count
    items0 = set()
    for x in data[:20000]:
        items0.update(x["sequence"])
    # not exact; we'll compute exact later to avoid huge memory.
    print("  (skip exact unique item count in full data; too big)")

    k = args.k
    for it in range(1, args.max_iter + 1):
        # 1) count item freq
        item_cnt = Counter()
        for x in data:
            item_cnt.update(x["sequence"])

        # 2) determine kept items
        keep_items = {i for i,c in item_cnt.items() if c >= k}

        # 3) filter sequences by kept items
        new_data = []
        for x in data:
            seq = [t for t in x["sequence"] if t in keep_items]
            if len(seq) >= k:  # user core
                new_data.append({"user_id": x["user_id"], "sequence": seq})

        if args.verbose:
            print(f"Iter {it}: users {len(data)} -> {len(new_data)}, items {len(keep_items)}")

        # convergence
        if len(new_data) == len(data):
            data = new_data
            print(f"Converged at iter={it}")
            break

        data = new_data

        if len(data) == 0:
            print("All data removed; k too large.")
            return

    # compute exact unique items after k-core
    item_set = set()
    for x in tqdm(data, desc="Collect unique items"):
        item_set.update(x["sequence"])

    U = len(data)
    I = len(item_set)
    interactions = sum(len(x["sequence"]) for x in data)
    dens = interactions / (U * I) if U > 0 and I > 0 else 0.0

    print("\n========================================")
    print(f"K-CORE DENSITY REPORT (k={k})")
    print("========================================")
    print(f"Users (U):         {U}")
    print(f"Items (I):         {I}")
    print(f"Interactions:      {interactions}")
    print(f"Matrix Capacity:   {U*I}")
    print(f"Density:           {dens:.8f}")
    print("========================================")

if __name__ == "__main__":
    main()
