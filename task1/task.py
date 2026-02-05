from typing import Tuple, List
from collections import defaultdict, deque


def main(s: str, e: str) -> Tuple[List[List[bool]], List[List[bool]], List[List[bool]], List[List[bool]], List[List[bool]]]:
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
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    r1 = [[False] * n for _ in range(n)]
    r2 = [[False] * n for _ in range(n)]
    r3 = [[False] * n for _ in range(n)]
    r4 = [[False] * n for _ in range(n)]
    r5 = [[False] * n for _ in range(n)]

    for parent, child_list in children.items():
        p_idx = node_to_idx[parent]
        for child in child_list:
            c_idx = node_to_idx[child]
            r1[p_idx][c_idx] = True

    for child, parent in parents.items():
        c_idx = node_to_idx[child]
        p_idx = node_to_idx[parent]
        r2[c_idx][p_idx] = True

    def get_all_descendants(node):
        descendants = set()
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for child in children[current]:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        return descendants

    for node in nodes:
        all_desc = get_all_descendants(node)
        node_idx = node_to_idx[node]
        for desc in all_desc:
            desc_idx = node_to_idx[desc]
            if desc not in children[node]:
                r3[node_idx][desc_idx] = True

    def get_all_ancestors(node):
        ancestors = set()
        current = node
        while current in parents:
            parent = parents[current]
            ancestors.add(parent)
            current = parent
        return ancestors

    for node in nodes:
        all_anc = get_all_ancestors(node)
        node_idx = node_to_idx[node]
        for anc in all_anc:
            anc_idx = node_to_idx[anc]
            if node not in children[anc]:
                r4[node_idx][anc_idx] = True

    for parent, child_list in children.items():
        for i, child1 in enumerate(child_list):
            for j, child2 in enumerate(child_list):
                if i != j:
                    c1_idx = node_to_idx[child1]
                    c2_idx = node_to_idx[child2]
                    r5[c1_idx][c2_idx] = True

    return (r1, r2, r3, r4, r5)


if __name__ == "__main__":
    input = "1,2\n1,3\n3,4\n3,5\n5,6\n6,7"
    root = "1"

    print(main(input, root))