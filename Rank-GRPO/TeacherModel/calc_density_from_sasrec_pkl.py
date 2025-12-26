import argparse
import pickle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True, help="Path to sasrec_dataset.pkl")
    ap.add_argument("--exclude_zero", action="store_true", help="Exclude 0 paddings if present in sequences")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        pkg = pickle.load(f)

    data = pkg["data"]          # list of {user_id, sequence}
    n_items = int(pkg["n_items"])  # your item id space size (usually max_id)

    n_users = len(data)

    interactions = 0
    for x in data:
        seq = x.get("sequence", [])
        if args.exclude_zero:
            interactions += sum(1 for t in seq if int(t) != 0)
        else:
            interactions += len(seq)

    # 如果你的 n_items 包含 padding=0 的槽位，而真实 item 是 1..n_items
    # 那么 I 就是 n_items（不含 0），这跟你训练用法一致即可
    I = n_items
    U = n_users

    density = interactions / (U * I) if U > 0 and I > 0 else 0.0

    print("========================================")
    print("DENSITY REPORT (from sasrec_dataset.pkl)")
    print("========================================")
    print(f"Users (U):         {U}")
    print(f"Items (I):         {I}")
    print(f"Interactions:      {interactions}")
    print(f"Matrix Capacity:   {U * I}")
    print(f"Density:           {density:.12f}")
    print("========================================")

if __name__ == "__main__":
    main()
