#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print("PROJECT_ROOT :", PROJECT_ROOT)
print("SRC_DIR      :", SRC_DIR)
print("DATA_DIR     :", DATA_DIR)

from io_utils import list_csv_datasets, read_transactions_csv
from brute_force import mine_frequent_itemsets, generate_rules

print("Datasets:", list_csv_datasets(DATA_DIR))


#  select dataset and threshold

# In[5]:


DATASET = "netflix.csv"     
MIN_SUPPORT = 0.20           
MIN_CONFIDENCE = 0.50       

path = os.path.join(DATA_DIR, DATASET)
tx = read_transactions_csv(path)

print(f"Dataset selected: {DATASET}")
print(f"Minimum support: {MIN_SUPPORT}")
print(f"Minimum confidence: {MIN_CONFIDENCE}")
print(f"Loaded {len(tx)} transactions from {DATASET}")


# Brute Force

# In[6]:


import time, pandas as pd
from IPython.display import display

t0 = time.time()
freq = mine_frequent_itemsets(tx, MIN_SUPPORT)
t1 = time.time()
rules = generate_rules(freq, tx, MIN_CONFIDENCE)
t2 = time.time()

print(f"Dataset: {DATASET}")
print(f"Transactions: {len(tx)}")
print(f"Min support: {MIN_SUPPORT}, Min confidence: {MIN_CONFIDENCE}")
print(f"Frequent itemsets: {len(freq)}  (mined in {t1 - t0:.4f}s)")
print(f"Rules: {len(rules)}  (generated in {t2 - t1:.4f}s)")

def df_itemsets(frequent):
    return pd.DataFrame([
        {"size": len(fs), "itemset": ", ".join(sorted(fs)), "support": round(s,4)}
        for fs, s in frequent
    ]).sort_values(["size","support"], ascending=[True, False]).reset_index(drop=True)

def df_rules(rules_list):
    return pd.DataFrame([
        {"LHS": ", ".join(sorted(L)), "RHS": ", ".join(sorted(R)), "support(X)": round(s,4), "confidence": round(c,4)}
        for L,R,s,c in rules_list
    ]).sort_values(["confidence","support(X)"], ascending=[False, False]).reset_index(drop=True)

itemsets_df = df_itemsets(freq)
rules_df = df_rules(rules)

print("\nTop frequent itemsets:")
display(itemsets_df.head(20))
print("\nTop rules:")
display(rules_df.head(20))


#  Apriori

# In[7]:


try:
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    using_apriori = True
except Exception:
    using_apriori = False
    print("Apriori unavailable (mlxtend not installed). Install:  python -m pip install mlxtend pandas")

if using_apriori:
    te = TransactionEncoder()
    arr = te.fit([list(t) for t in tx]).transform([list(t) for t in tx])
    df = pd.DataFrame(arr, columns=te.columns_)

    A_itemsets = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
    A_rules = association_rules(A_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

    from IPython.display import display
    print("Apriori — frequent itemsets (top 20):")
    display(A_itemsets.sort_values(['support','itemsets'], ascending=[False, True]).reset_index(drop=True).head(20))

    print("\nApriori — rules (top 20):")
    display(A_rules.sort_values(['confidence','support'], ascending=[False, False]).reset_index(drop=True)[['antecedents','consequents','support','confidence','lift']].head(20))


#  FP‑Growth

# In[8]:


try:
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    using_fpg = True
except Exception:
    using_fpg = False
    print("FP-Growth unavailable (mlxtend not installed). Install:  python -m pip install mlxtend pandas")

if using_fpg:
    te = TransactionEncoder()
    arr = te.fit([list(t) for t in tx]).transform([list(t) for t in tx])
    df = pd.DataFrame(arr, columns=te.columns_)

    F_itemsets = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True)
    F_rules = association_rules(F_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

    from IPython.display import display
    print("FP-Growth — frequent itemsets (top 20):")
    display(F_itemsets.sort_values(['support','itemsets'], ascending=[False, True]).reset_index(drop=True).head(20))

    print("\nFP-Growth — rules (top 20):")
    display(F_rules.sort_values(['confidence','support'], ascending=[False, False]).reset_index(drop=True)[['antecedents','consequents','support','confidence','lift']].head(20))

