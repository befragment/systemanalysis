from typing import Tuple, List
from collections import defaultdict, deque
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task1.task import main as compute_relations


def main(s: str, e: str) -> Tuple[float, float]:
    edges = []
    if s.strip():
        for line in s.strip().split('\n'):
            if line.strip():
                parent, child = line.strip().split(',')
                edges.append((parent.strip(), child.strip()))

    children = defaultdict(list)
    parents = {}
    all_nodes = set([e])

    for parent, child in edges:
        children[parent].append(child)
        parents[child] = parent
        all_nodes.add(parent)
        all_nodes.add(child)

    nodes = sorted(all_nodes)

    relations = compute_relations(s, e)
    r1, r2, r3, r4, r5 = relations

    n = len(nodes)
    k = 5

    l = [[0] * n for _ in range(k)]

    for relation_idx, relation_matrix in enumerate(relations):
        for node_idx in range(n):
            l[relation_idx][node_idx] = sum(1 for val in relation_matrix[node_idx] if val)

    max_connections = n - 1
    total_entropy = 0.0

    for relation_idx in range(k):
        for node_idx in range(n):
            l_ij = l[relation_idx][node_idx]
            if l_ij > 0:
                P = l_ij / max_connections
                H_partial = -P * math.log2(P)
                total_entropy += H_partial

    c = 1.0 / (math.e * math.log(2))
    H_ref = c * n * k

    normalized_complexity = total_entropy / H_ref

    entropy_rounded = round(total_entropy, 1)
    complexity_rounded = round(normalized_complexity, 1)

    return (entropy_rounded, complexity_rounded)


if __name__ == "__main__":
    test_input = "1,2\n1,3\n3,4\n3,5"
    root = "1"

    entropy, normalized = main(test_input, root)
    print(entropy, normalized)