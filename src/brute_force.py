from itertools import combinations

def support_count(itemset, transactions):
    return sum(1 for t in transactions if itemset.issubset(t))

def support_fraction(itemset, transactions):
    n = len(transactions)
    return support_count(itemset, transactions) / n if n else 0.0

def all_items(transactions):
    s = set()
    for t in transactions: s |= set(t)
    return sorted(s)

def enumerate_k(items, k):
    return [frozenset(c) for c in combinations(items, k)]

def mine_frequent_itemsets(transactions, min_support):
    items = all_items(transactions)
    out = []
    k = 1
    while True:
        level = []
        for cand in enumerate_k(items, k):
            sup = support_fraction(cand, transactions)
            if sup >= min_support:
                level.append((cand, sup))
        if not level: break
        out.extend(level)
        k += 1
    out.sort(key=lambda x: (len(x[0]), -x[1], sorted(list(x[0]))))
    return out

def make_support_lookup(freq):
    return {fs: sup for fs, sup in freq}

def generate_rules(freq, transactions, min_conf):
    sup_map = make_support_lookup(freq)
    rules = []
    for X, sX in freq:
        if len(X) < 2: continue
        items = list(X)
        for r in range(1, len(items)):
            for lhs in combinations(items, r):
                L = frozenset(lhs)
                R = X - L
                sL = sup_map.get(L)
                if sL is None:
                    sL = support_fraction(L, transactions)
                conf = sX / sL if sL else 0.0
                if conf >= min_conf:
                    rules.append((L, R, sX, conf))
    rules.sort(key=lambda x: (-x[3], -x[2], len(x[0]), sorted(list(x[0]))))
    return rules
