def main(v: str) -> tuple[tuple[int | bool]]:
    v = v.split('\n')
    edges = []
    for e in v:
        e = e.split('\t')
        e = [int(x) for x in e]
        edges.append(e)
    print(edges)
    flat = [item for sublist in edges for item in sublist]
    unique_vertexes = list(set(flat))
    n = len(unique_vertexes)

    matrix = [[0 for _ in range(n)] for _ in range(n)]

    # assuming matrix top left corner has indecies [1, 1]
    for v1, v2 in edges:
        matrix[v1 - 1][v2 - 1] = 1
        matrix[v2 - 1][v1 - 1] = 1
    
    return matrix


if __name__ == '__main__':
    t = open('file.csv').read()
    result = main(t)
    for row in result:
        print(row)