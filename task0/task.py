def main(v: str) -> list[list[int | bool]]:
    v = v.split('\n')
    edges = []
    for e in v:
        edges.append(e.split('\t'))

    unique_vertexes = set([item for sublist in edges for item in sublist])
    n = len(unique_vertexes)
    edges_mapping = {
        x: int(x) - 1 for x in unique_vertexes
    }

    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for v1, v2 in edges:
        v1, v2 = edges_mapping[v1], edges_mapping[v2]
        matrix[v1][v2] = 1
        matrix[v2][v1] = 1
    
    return matrix


if __name__ == '__main__':
    t = open('file.csv').read()
    result = main(t)
    for row in result:
        print(row)