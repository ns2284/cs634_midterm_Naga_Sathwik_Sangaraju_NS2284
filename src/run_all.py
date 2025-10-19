import os
from io_utils import (
    list_csv_datasets, read_transactions_csv, ensure_out_dirs,
    safe_num, ask_one_dataset, ask_float
)
from brute_force import mine_frequent_itemsets, generate_rules
from apriori_lib import run_apriori
from fpgrowth_lib import run_fpgrowth

def fmt_set(S):
    return "{" + ", ".join(sorted(map(str, S))) + "}"

def print_brute_results(dataset, tx, freq, rules, minsup, minconf):
    print(f"\n=== [Brute Force] {dataset} ===")
    print(f"Transactions: {len(tx)} | MinSupport={minsup}, MinConfidence={minconf}")
    print(f"\nFrequent Itemsets ({len(freq)} total):")
    for fs, sup in freq:
        print(f"  {fmt_set(fs):<40} support={sup:.4f}")
    print(f"\nAssociation Rules ({len(rules)} total):")
    for L, R, s, c in rules:
        print(f"  {fmt_set(L)} -> {fmt_set(R)}  support={s:.4f}  confidence={c:.4f}")

def print_df_itemsets(title, df):
    print(f"\n{title} ({len(df)} total):")
    for _, row in df.iterrows():
        items = fmt_set(row["itemsets"])
        print(f"  {items:<40} support={float(row['support']):.4f}")

def print_df_rules(title, df):
    print(f"\n{title} ({len(df)} total):")
    for _, r in df.iterrows():
        a = fmt_set(r["antecedents"])
        c = fmt_set(r["consequents"])
        print(f"  {a} -> {c}  support={float(r['support']):.4f}  confidence={float(r['confidence']):.4f}  lift={float(r.get('lift', 0.0)):.4f}")

def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

    names = list_csv_datasets(DATA_DIR)
    if not names:
        print(f"No .csv datasets found in {DATA_DIR}")
        return

    dataset = ask_one_dataset(names)
    minsup = ask_float("minimum support")
    minconf = ask_float("minimum confidence")

    path = os.path.join(DATA_DIR, dataset)
    tx = read_transactions_csv(path)

    # BRUTE FORCE 
    freq = mine_frequent_itemsets(tx, minsup)
    rules = generate_rules(freq, tx, minconf)
    print_brute_results(dataset, tx, freq, rules, minsup, minconf)

    #  APRIORI 
    L, R = run_apriori(tx, minsup, minconf)
    print(f"\n=== [Apriori] {dataset} ===")
    print_df_itemsets("Frequent Itemsets", L)
    print_df_rules("Association Rules", R)

    # FP-GROWTH 
    L2, R2 = run_fpgrowth(tx, minsup, minconf)
    print(f"\n=== [FP-Growth] {dataset} ===")
    print_df_itemsets("Frequent Itemsets", L2)
    print_df_rules("Association Rules", R2)

if __name__ == "__main__":
    main()
