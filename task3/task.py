import json
from typing import Union

Atom = Union[int, str]
Tier = Union[Atom, list[Atom]]
RankSpec = list[Tier]


def flatten_rank(spec: RankSpec) -> list[Atom]:
    flat: list[Atom] = []
    for node in spec:
        if isinstance(node, list):
            flat += node
        else:
            flat.append(node)
    return flat


def index_levels(spec: RankSpec) -> dict[Atom, int]:
    level_of: dict[Atom, int] = {}
    for level, node in enumerate(spec):
        if isinstance(node, list):
            for v in node:
                level_of[v] = level
        else:
            level_of[node] = level
    return level_of


def strict_pref_graph(items: list[Atom], level_of: dict[Atom, int]) -> dict[Atom, dict[Atom, int]]:
    g: dict[Atom, dict[Atom, int]] = {}
    for a in items:
        row: dict[Atom, int] = {}
        for b in items:
            row[b] = 1 if level_of[a] < level_of[b] else 0
        g[a] = row
    return g


def transpose_graph(g: dict[Atom, dict[Atom, int]], items: list[Atom]) -> dict[Atom, dict[Atom, int]]:
    gt: dict[Atom, dict[Atom, int]] = {}
    for a in items:
        gt[a] = {b: g[b][a] for b in items}
    return gt


def bit_and(a: dict[Atom, dict[Atom, int]], b: dict[Atom, dict[Atom, int]], items: list[Atom]) -> dict[Atom, dict[Atom, int]]:
    out: dict[Atom, dict[Atom, int]] = {}
    for i in items:
        out[i] = {j: (a[i][j] & b[i][j]) for j in items}
    return out


def bit_or(a: dict[Atom, dict[Atom, int]], b: dict[Atom, dict[Atom, int]], items: list[Atom]) -> dict[Atom, dict[Atom, int]]:
    out: dict[Atom, dict[Atom, int]] = {}
    for i in items:
        out[i] = {j: (a[i][j] | b[i][j]) for j in items}
    return out


def conflict_pairs(g1: dict[Atom, dict[Atom, int]], g2: dict[Atom, dict[Atom, int]], items: list[Atom]) -> list[list[Atom]]:
    pairs: list[list[Atom]] = []
    m = len(items)
    for p in range(m):
        for q in range(p + 1, m):
            x, y = items[p], items[q]
            if (g1[x][y] == 1 and g2[y][x] == 1) or (g1[y][x] == 1 and g2[x][y] == 1):
                pairs.append([x, y])
    return pairs


def transitive_closure(rel: dict[Atom, dict[Atom, int]], items: list[Atom]) -> dict[Atom, dict[Atom, int]]:
    clo: dict[Atom, dict[Atom, int]] = {i: {j: rel[i][j] for j in items} for i in items}
    for k in items:
        for i in items:
            ik = clo[i][k]
            if ik == 0:
                continue
            for j in items:
                clo[i][j] = clo[i][j] | (ik & clo[k][j])
    return clo


def components_from_conflicts(items: list[Atom], pairs: list[list[Atom]]) -> list[list[Atom]]:
    adj: dict[Atom, dict[Atom, int]] = {i: {j: (1 if i == j else 0) for j in items} for i in items}
    for x, y in pairs:
        adj[x][y] = 1
        adj[y][x] = 1

    reach = transitive_closure(adj, items)

    seen: set[Atom] = set()
    groups: list[list[Atom]] = []
    for root in items:
        if root in seen:
            continue
        group: list[Atom] = []
        for v in items:
            if reach[root][v] == 1:
                group.append(v)
                seen.add(v)
        if group:
            groups.append(group)
    return groups


def consensus_nonstrict(g1: dict[Atom, dict[Atom, int]], g2: dict[Atom, dict[Atom, int]], items: list[Atom]) -> dict[Atom, dict[Atom, int]]:
    # i не хуже j, если j не строго лучше i (одновременно в обеих ранжировках)
    c: dict[Atom, dict[Atom, int]] = {}
    for i in items:
        c[i] = {}
        for j in items:
            ok1 = 1 - g1[j][i]
            ok2 = 1 - g2[j][i]
            c[i][j] = ok1 & ok2
    return c


def sort_groups(groups: list[list[Atom]], a_level: dict[Atom, int], b_level: dict[Atom, int]) -> list[list[Atom]]:
    def mean_level(group: list[Atom]) -> float:
        s = sum(a_level[v] + b_level[v] for v in group)
        return s / (2 * len(group))

    return sorted(groups, key=mean_level)


def pack_rank(groups: list[list[Atom]]) -> RankSpec:
    out: RankSpec = []
    for g in groups:
        g_sorted = sorted(g, key=lambda x: (isinstance(x, str), x))
        out.append(g_sorted[0] if len(g_sorted) == 1 else g_sorted)
    return out


def main(rank_json_left: str, rank_json_right: str) -> str:
    left: RankSpec = json.loads(rank_json_left)
    right: RankSpec = json.loads(rank_json_right)

    universe = flatten_rank(left)
    left_level = index_levels(left)
    right_level = index_levels(right)

    pref_left = strict_pref_graph(universe, left_level)
    pref_right = strict_pref_graph(universe, right_level)

    core = conflict_pairs(pref_left, pref_right, universe)

    groups = components_from_conflicts(universe, core)
    groups_sorted = sort_groups(groups, left_level, right_level)
    consensus = pack_rank(groups_sorted)

    payload = {"core": core, "ranking": consensus}
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    a = '[1,[2,3],4,[5,6,7],8,9,10]'
    b = '[[1,2],[3,4,5],6,7,9,[8,10]]'

    out = main(a, b)
    print(f"Результат: {out}\n")
    print("Ожидаемое ядро противоречий: [[8, 9]]")
    print("Ожидаемая ранжировка:        [1, 2, 3, 4, 5, 6, 7, [8, 9], 10]")