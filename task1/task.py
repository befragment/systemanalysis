from typing import List, Tuple
import csv


def make_bool_matrix(size: int) -> List[List[bool]]:
    return [[False for _ in range(size)] for _ in range(size)]


def copy_matrix(mat: List[List[bool]]) -> List[List[bool]]:
    return [row.copy() for row in mat]


def matrix_transpose(mat: List[List[bool]]) -> List[List[bool]]:
    n = len(mat)
    transposed = make_bool_matrix(n)
    for r in range(n):
        for c in range(n):
            transposed[c][r] = mat[r][c]
    return transposed


def floyd_warshall_reachability(adj: List[List[bool]]) -> List[List[bool]]:
    n = len(adj)
    reachable = copy_matrix(adj)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if reachable[i][k] and reachable[k][j]:
                    reachable[i][j] = True
    return reachable


def bfs_levels(start: int, adj: List[List[bool]]) -> List[int]:
    n = len(adj)
    level = [-1] * n
    queue = [start]
    level[start] = 0

    while queue:
        cur = queue.pop(0)
        for nxt in range(n):
            if adj[cur][nxt] and level[nxt] == -1:
                level[nxt] = level[cur] + 1
                queue.append(nxt)

    return level


def main(src_file: str, root_vertex: str) -> Tuple[
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]]
]:
    edges: List[Tuple[int, int]] = []
    with open(src_file, "r") as f:
        for a, b in csv.reader(f):
            edges.append((int(a), int(b)))

    vertices = sorted({v for a, b in edges for v in (a, b)})
    n = len(vertices)

    idx_map = {v: i for i, v in enumerate(vertices)}

    adj = make_bool_matrix(n)
    for a, b in edges:
        adj[idx_map[a]][idx_map[b]] = True

    r1 = copy_matrix(adj)

    r2 = matrix_transpose(r1)

    full_reach = floyd_warshall_reachability(adj)
    r3 = make_bool_matrix(n)
    for i in range(n):
        for j in range(n):
            if full_reach[i][j] and not adj[i][j]:
                r3[i][j] = True

    r4 = matrix_transpose(r3)

    start_idx = idx_map[int(root_vertex)]
    lvl = bfs_levels(start_idx, adj)
    r5 = make_bool_matrix(n)

    for i in range(n):
        for j in range(n):
            if i != j and lvl[i] != -1 and lvl[i] == lvl[j]:
                r5[i][j] = True

    return r1, r2, r3, r4, r5


if __name__ == '__main__':
    inp = "1,2\n1,3\n3,4\n3,5\n5,6\n6,7"
    root = "1"

    print(main(inp, root))