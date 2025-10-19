import pandas as pd
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
except Exception:
    raise SystemExit("mlxtend is required. Try: pip install mlxtend pandas")

def run_apriori(transactions, min_support, min_confidence):
    te = TransactionEncoder()
    arr = te.fit([list(t) for t in transactions]).transform([list(t) for t in transactions])
    df = pd.DataFrame(arr, columns=te.columns_)
    L = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(L, metric='confidence', min_threshold=min_confidence)
    return L, rules
