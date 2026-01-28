import json
from math import isclose
from typing import Any, Final

EPS: Final = 1e-9


def log_if(enabled: bool, *args: object) -> None:
    if enabled:
        print("[DEBUG]", *args)


def decode_lv(lv_json: Any) -> dict[str, list[tuple[float, float]]]:
    """
    Принимает LV как str/dict/list и возвращает:
      { term_id: [(x1,y1), (x2,y2), ...] }  (отсортировано по x)
    """
    payload = json.loads(lv_json) if isinstance(lv_json, str) else lv_json

    terms: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                terms = v
                break
    elif isinstance(payload, list):
        terms = payload

    out: dict[str, list[tuple[float, float]]] = {}
    for term in terms:
        term_id = str(term.get("id"))
        raw_pts = term.get("points", []) or []
        pts = sorted(((float(x), float(y)) for x, y in raw_pts), key=lambda p: p[0])
        out[term_id] = pts
    return out


def membership(points: list[tuple[float, float]], x_val: float) -> float:
    if not points:
        return 0.0
    x_val = float(x_val)

    if x_val <= points[0][0]:
        return float(points[0][1])
    if x_val >= points[-1][0]:
        return float(points[-1][1])

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= x_val <= x2:
            if isclose(x2, x1):
                return float(y2)
            t = (x_val - x1) / (x2 - x1)
            return float(y1 + (y2 - y1) * t)
    return 0.0


def normalize_rules(rules_input: Any) -> list[tuple[str, str]]:
    data = json.loads(rules_input) if isinstance(rules_input, str) else rules_input
    normalized: list[tuple[Any, Any]] = []

    if isinstance(data, dict):
        for key, val in data.items():
            inputs = [k.strip() for k in str(key).split(",") if k.strip()]
            for inp in inputs:
                normalized.append((inp, val))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append((item[0], item[1]))
            elif isinstance(item, dict):
                if "if" in item and "then" in item:
                    normalized.append((item["if"], item["then"]))
                elif "from" in item and "to" in item:
                    normalized.append((item["from"], item["to"]))
                elif "input" in item and "output" in item:
                    normalized.append((item["input"], item["output"]))
                else:
                    keys = list(item.keys())
                    if len(keys) >= 2:
                        normalized.append((item[keys[0]], item[keys[1]]))

    return [(str(a), str(b)) for a, b in normalized]


def find_x_intersect(p1: tuple[float, float], p2: tuple[float, float], level: float) -> float | None:
    x1, y1 = p1
    x2, y2 = p2
    if isclose(y2, y1):
        return None
    t = (level - y1) / (y2 - y1)
    if -1e-12 <= t <= 1 + 1e-12:
        return x1 + t * (x2 - x1)
    return None


def clip_shape(points: list[tuple[float, float]], level: float, eps: float = EPS) -> list[tuple[float, float]]:
    if not points:
        return []
    level = float(level)

    # 1) ограничиваем по y: y := min(y, level) + добавляем точки пересечения
    tmp: list[tuple[float, float]] = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        tmp.append((float(p1[0]), min(float(p1[1]), level)))

        if (p1[1] - level) * (p2[1] - level) < -eps:
            ix = find_x_intersect(p1, p2, level)
            if ix is not None:
                tmp.append((float(ix), level))

    last = points[-1]
    tmp.append((float(last[0]), min(float(last[1]), level)))

    tmp.sort(key=lambda p: p[0])

    # 2) схлопываем одинаковые x, берём max y
    uniq: list[tuple[float, float]] = []
    for x, y in tmp:
        if uniq and isclose(uniq[-1][0], x, abs_tol=eps):
            uniq[-1] = (uniq[-1][0], max(uniq[-1][1], y))
        else:
            uniq.append((x, y))

    # 3) убираем "дробление" горизонтали на уровне level
    final: list[tuple[float, float]] = []
    i = 0
    n = len(uniq)
    while i < n:
        x, y = uniq[i]
        if isclose(y, level, abs_tol=eps):
            j = i
            while j + 1 < n and isclose(uniq[j + 1][1], level, abs_tol=eps):
                j += 1
            final.append((x, level))
            if not isclose(uniq[j][0], x, abs_tol=eps):
                final.append((uniq[j][0], level))
            i = j + 1
        else:
            final.append((x, y))
            i += 1

    # 4) убираем дубликаты точек
    out: list[tuple[float, float]] = []
    for p in final:
        if out and isclose(out[-1][0], p[0], abs_tol=1e-12) and isclose(out[-1][1], p[1], abs_tol=1e-12):
            continue
        out.append(p)
    return out


def compute_input_memberships(inputs_map: dict[str, list[tuple[float, float]]], x: float) -> dict[str, float]:
    return {name: membership(pts, x) for name, pts in inputs_map.items()}


def aggregate_output_levels(
    rule_pairs: list[tuple[str, str]],
    in_mu: dict[str, float],
    out_terms: list[str],
) -> dict[str, float]:
    levels: dict[str, float] = {k: 0.0 for k in out_terms}
    for in_term, out_term in rule_pairs:
        deg = float(in_mu.get(in_term, 0.0))
        if deg > levels.get(out_term, 0.0):
            levels[out_term] = deg
    return levels


def clip_outputs(
    outputs_map: dict[str, list[tuple[float, float]]],
    levels: dict[str, float],
    eps: float = EPS,
) -> dict[str, list[tuple[float, float]]]:
    clipped: dict[str, list[tuple[float, float]]] = {}
    for name, pts in outputs_map.items():
        a = float(levels.get(name, 0.0))
        clipped[name] = [] if a <= eps else clip_shape(pts, a, eps=eps)
    return clipped


def global_max_height(clipped: dict[str, list[tuple[float, float]]]) -> float:
    ys = [y for shape in clipped.values() for (_, y) in shape]
    return float(max(ys)) if ys else 0.0


def fallback_centroid(outputs_map: dict[str, list[tuple[float, float]]]) -> float:
    centers: list[float] = []
    for pts in outputs_map.values():
        if pts:
            centers.append((pts[0][0] + pts[-1][0]) / 2.0)
    return float(sum(centers) / len(centers)) if centers else 0.0


def extract_max_intervals(
    clipped: dict[str, list[tuple[float, float]]],
    ymax: float,
    tol: float = 1e-7,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for pts in clipped.values():
        if not pts:
            continue
        i = 0
        n = len(pts)
        while i < n:
            if isclose(pts[i][1], ymax, abs_tol=tol):
                j = i
                while j + 1 < n and isclose(pts[j + 1][1], ymax, abs_tol=tol):
                    j += 1
                intervals.append((float(pts[i][0]), float(pts[j][0])))
                i = j + 1
            else:
                i += 1
    return intervals


def merge_intervals(intervals: list[tuple[float, float]], eps: float = EPS) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda t: t[0])
    merged: list[list[float]] = []
    for l, r in intervals_sorted:
        if not merged:
            merged.append([l, r])
        else:
            last = merged[-1]
            if l <= last[1] + eps:
                last[1] = max(last[1], r)
            else:
                merged.append([l, r])
    return [(float(a), float(b)) for a, b in merged]


def centroid_of_max_plateau(
    intervals: list[tuple[float, float]],
    clipped: dict[str, list[tuple[float, float]]],
    ymax: float,
    eps: float = EPS,
) -> float:
    # Длина плато
    total_len = sum(max(0.0, r - l) for l, r in intervals)
    if total_len <= eps:
        mids = [(l + r) / 2.0 for l, r in intervals]
        if mids:
            return float(sum(mids) / len(mids))

        xs_exact: list[float] = []
        for pts in clipped.values():
            for x, y in pts:
                if isclose(y, ymax, abs_tol=1e-7):
                    xs_exact.append(float(x))
        return float(sum(xs_exact) / len(xs_exact)) if xs_exact else 0.0

    moment_sum = 0.0
    for l, r in intervals:
        moment_sum += (r * r - l * l)

    return float(0.5 * moment_sum / total_len)

def run_inference(
    lv_input: Any,
    lv_output: Any,
    rules: Any,
    x: float,
    verbose: bool = False,
) -> float:
    in_terms = decode_lv(lv_input)
    out_terms = decode_lv(lv_output)
    rule_pairs = normalize_rules(rules)

    mu_in = compute_input_memberships(in_terms, x)

    log_if(verbose, "T =", float(x))
    log_if(verbose, "Степени входа =", json.dumps(mu_in, ensure_ascii=False))

    alpha_out = aggregate_output_levels(rule_pairs, mu_in, list(out_terms.keys()))
    log_if(verbose, "Alpha уровни выходов =", json.dumps(alpha_out, ensure_ascii=False))

    clipped = clip_outputs(out_terms, alpha_out, eps=EPS)

    if verbose:
        for name, pts in clipped.items():
            if not pts:
                log_if(verbose, f"Clipping[{name}]: пропущен (0)")
            else:
                dbg_pts = [(round(px, 6), round(py, 6)) for px, py in pts]
                log_if(verbose, f"Clipping[{name}]:", dbg_pts)

    ymax = global_max_height(clipped)
    log_if(verbose, "Global Max Y =", round(ymax, 6))

    if ymax <= EPS:
        fb = fallback_centroid(out_terms)
        log_if(verbose, "Fallback centroid =", round(fb, 6))
        return float(fb)

    raw_intervals = extract_max_intervals(clipped, ymax)
    log_if(verbose, "Raw max intervals =", [(round(l, 6), round(r, 6)) for l, r in raw_intervals])

    merged = merge_intervals(raw_intervals, eps=EPS)
    log_if(verbose, "Merged intervals =", [(round(l, 6), round(r, 6)) for l, r in merged])

    total_len = sum(max(0.0, r - l) for l, r in merged)
    log_if(verbose, "Total Length L =", round(total_len, 6))

    res = centroid_of_max_plateau(merged, clipped, ymax, eps=EPS)
    log_if(verbose, "Result X_opt =", round(res, 6))
    return float(res)

if __name__ == "__main__":
    print(
        run_inference(
            '''
{
    "температура": [
        {
            "id": "холодно",
            "points": [
                [0,1],
                [18,1],
                [22,0],
                [50,0]
            ]
        },
        {
            "id": "комфортно",
            "points": [
                [18,0],
                [22,1],
                [24,1],
                [26,0]
            ]
        },
        {
            "id": "жарко",
            "points": [
                [0,0],
                [24,0],
                [26,1],
                [50,1]
            ]
        }
    ]
}
''',
            '''
{
  "управление": [
      {
        "id": "слабо",
        "points": [
            [0,0],
            [0,1],
            [5,1],
            [8,0]
        ]
      },
      {
        "id": "умеренно",
        "points": [
            [5,0],
            [8,1],
            [13,1],
            [16,0]
        ]
      },
      {
        "id": "интенсивно",
        "points": [
            [13,0],
            [18,1],
            [23,1],
            [26,0]
        ]
      }
  ]
}
''',
            '''
{
  "холодно": "интенсивно",
  "комфортно": "умеренно",
  "жарко": "слабо"
}
''',
            25,
            verbose=True,
        )
    )