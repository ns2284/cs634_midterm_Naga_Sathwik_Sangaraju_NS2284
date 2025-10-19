import os, csv

def list_csv_datasets(data_dir):
    return sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.csv')])

def read_transactions_csv(path):
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            parts = [p.strip() for p in row['items'].split(',') if p.strip()]
            out.append(frozenset(parts))
    return out

def ensure_out_dirs(root_out, dataset):
    base = os.path.join(root_out, dataset.replace('.csv',''))
    paths = {
        'base': base,
        'brute': os.path.join(base, 'brute_force'),
        'apriori': os.path.join(base, 'apriori'),
        'fpgrowth': os.path.join(base, 'fpgrowth'),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def safe_num(x):
    return str(x).replace('.', 'p')

def ask_one_dataset(names):
    """Let the user choose a dataset by number instead of typing its name."""
    print("Available datasets:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    raw = input("Enter the number of the dataset you want: ").strip()
    if not raw.isdigit():
        print(f"Please enter a valid number (1–{len(names)})"); exit(1)
    idx = int(raw)
    if idx < 1 or idx > len(names):
        print(f"Invalid selection. Please pick a number between 1 and {len(names)}."); exit(1)
    selected = names[idx - 1]
    print(f"You selected: {selected}")
    return selected

def ask_float(label):
    """Ask for a float in [0,1]."""
    raw = input(f"Enter {label} (0..1): ").strip()
    try:
        val = float(raw)
    except Exception:
        print(f"{label} must be a number like 0.2"); exit(1)
    if not (0.0 <= val <= 1.0):
        print(f"{label} must be between 0 and 1"); exit(1)
    return val
