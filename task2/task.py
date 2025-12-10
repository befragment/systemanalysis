from typing import List, Tuple
import csv
import math


def make_matrix(size: int) -> List[List[bool]]:
    return [[False for _ in range(size)] for _ in range(size)]


def copy_mat(mat: List[List[bool]]) -> List[List[bool]]:
    return [row.copy() for row in mat]


def mat_transpose(mat: List[List[bool]]) -> List[List[bool]]:
    n = len(mat)
    tr = make_matrix(n)
    for r in range(n):
        for c in range(n):
            tr[c][r] = mat[r][c]
    return tr


def compute_reachability(graph: List[List[bool]]) -> List[List[bool]]:
    n = len(graph)
    reach = copy_mat(graph)

    for mid in range(n):
        for a in range(n):
            for b in range(n):
                if reach[a][mid] and reach[mid][b]:
                    reach[a][b] = True

    return reach


def bfs_distances(start: int, graph: List[List[bool]]) -> List[int]:
    n = len(graph)
    dist = [-1] * n
    queue = [start]
    dist[start] = 0

    while queue:
        v = queue.pop(0)
        for u in range(n):
            if graph[v][u] and dist[u] == -1:
                dist[u] = dist[v] + 1
                queue.append(u)

    return dist


def build_relations(path: str, root_vertex: str) -> Tuple[
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]]
]:
    # читаем рёбра
    edge_list = []
    with open(path, "r") as f:
        for a, b in csv.reader(f):
            edge_list.append((int(a), int(b)))

    # вершины
    unique_nodes = sorted({x for a, b in edge_list for x in (a, b)})
    n = len(unique_nodes)
    node_index = {node: i for i, node in enumerate(unique_nodes)}

    # матрица смежности
    graph = make_matrix(n)
    for a, b in edge_list:
        graph[node_index[a]][node_index[b]] = True

    rel1 = copy_mat(graph)
    rel2 = mat_transpose(rel1)

    full_reach = compute_reachability(graph)

    rel3 = make_matrix(n)
    for i in range(n):
        for j in range(n):
            if full_reach[i][j] and not graph[i][j]:
                rel3[i][j] = True

    rel4 = mat_transpose(rel3)

    # уровни
    start_idx = node_index[int(root_vertex)]
    dist = bfs_distances(start_idx, graph)

    rel5 = make_matrix(n)
    for i in range(n):
        for j in range(n):
            if i != j and dist[i] != -1 and dist[i] == dist[j]:
                rel5[i][j] = True

    return rel1, rel2, rel3, rel4, rel5


def count_links(mat: List[List[bool]]) -> int:
    return sum(val for row in mat for val in row)


def entropy_value(mats: Tuple[List[List[bool]], ...]) -> float:
    n = len(mats[0])
    max_deg = n - 1
    if max_deg == 0:
        return 0.0

    entropy_total = 0.0

    for col in range(n):
        for matrix in mats:
            degree = sum(1 for x in range(n) if matrix[col][x])
            if degree:
                p = degree / max_deg
                entropy_total += -p * math.log2(p)

    return entropy_total


def complexity_score(mats: Tuple[List[List[bool]], ...], entropy: float) -> float:
    n = len(mats[0])
    layers = 5

    coef = 1 / (math.e * math.log(2))
    ref_entropy = coef * layers * n

    if ref_entropy == 0:
        return 0.0

    return entropy / ref_entropy


def task(file_path: str, start: str) -> Tuple[float, float]:
    relations = build_relations(file_path, start)

    ent = entropy_value(relations)
    comp = complexity_score(relations, ent)

    return round(ent, 1), round(comp, 2)