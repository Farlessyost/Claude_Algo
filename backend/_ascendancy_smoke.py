"""Test the Ulanowicz ascendancy/capacity/reserve math against known structures."""
from backend.ecology import ascendancy_metrics, information_network

def show(label, edges):
    m = ascendancy_metrics(edges)
    print(f"\n{label}")
    print(f"  edges        : {len(edges)} ({m['n_active_edges']} active)")
    print(f"  TST          : {m['TST']:.4f}")
    print(f"  AMI          : {m['AMI']:.4f}")
    print(f"  A (ascend)   : {m['A']:.4f}")
    print(f"  H (entropy)  : {m['H']:.4f}")
    print(f"  C (capacity) : {m['C']:.4f}")
    print(f"  R (reserve)  : {m['R']:.4f}")
    print(f"  rel_A        : {m['rel_ascendancy']:.4f}")
    print(f"  rel_R        : {m['rel_reserve']:.4f}")
    print(f"  rel_A + rel_R: {m['rel_ascendancy'] + m['rel_reserve']:.4f}  (expect ~1.0)")

# 1) One dominant edge — should give HIGH rel_ascendancy
show("[1] Single dominant edge (A->B = 1.0, others ~0)",
     {("A", "B"): 1.0, ("A", "C"): 0.05, ("B", "A"): 0.05, ("B", "C"): 0.05,
      ("C", "A"): 0.05, ("C", "B"): 0.05})

# 2) Uniform flow — should give LOW rel_ascendancy (high reserve)
show("[2] Uniform flow (all edges equal)",
     {("A", "B"): 0.5, ("A", "C"): 0.5, ("B", "A"): 0.5, ("B", "C"): 0.5,
      ("C", "A"): 0.5, ("C", "B"): 0.5})

# 3) Two competing pathways — moderate
show("[3] Two competing pathways (A->B and C->B dominant)",
     {("A", "B"): 0.8, ("C", "B"): 0.8, ("A", "C"): 0.1, ("B", "A"): 0.1,
      ("B", "C"): 0.1, ("C", "A"): 0.1})

# 4) Empty — degenerate
show("[4] Empty graph",
     {})

# 5) Realistic: 9 nodes, weighted edges (mimic a live ecology snapshot)
import random
random.seed(42)
nodes = ["price", "vol", "range", "volume", "eth_perp", "btc_spot",
          "eth_spot", "liq_proxy", "funding"]
edges = {}
for a in nodes:
    for b in nodes:
        if a == b:
            continue
        # Mostly weak (uniform), with a few strong directional links
        if (a, b) in (("price", "vol"), ("vol", "range"), ("btc_spot", "price")):
            edges[(a, b)] = random.uniform(0.4, 0.8)
        else:
            edges[(a, b)] = random.uniform(-0.2, 0.2)
show("[5] Realistic 9-node graph w/ 3 strong directional links", edges)
