import json
from math import isclose

def main(LVinput, LVoutput, rules, T, verbose=False):
    EPSILON = 1e-9

    def log_msg(*args):
        if verbose:
            print("[LOG]", *args)

    def decode_lv(json_data):
        payload = json.loads(json_data) if isinstance(json_data, str) else json_data

        term_items = []
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, list):
                    term_items = v
                    break
        elif isinstance(payload, list):
            term_items = payload

        var_points = {}
        for term_obj in term_items:
            term_id = term_obj.get('id')
            raw_pts = term_obj.get('points', [])
            sorted_pts = sorted([(float(a), float(b)) for a, b in raw_pts], key=lambda p: p[0])
            var_points[term_id] = sorted_pts
        return var_points

    def get_membership(pts, x_val):
        if not pts:
            return 0.0
        x_num = float(x_val)

        if x_num <= pts[0][0]:
            return float(pts[0][1])
        if x_num >= pts[-1][0]:
            return float(pts[-1][1])

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]

            if x1 <= x_num <= x2:
                if isclose(x2, x1):
                    return float(y2)
                k = (x_num - x1) / (x2 - x1)
                return float(y1 + (y2 - y1) * k)
        return 0.0

    def normalize_rules(rules_input):
        rules_data = json.loads(rules_input) if isinstance(rules_input, str) else rules_input
        pairs = []

        if isinstance(rules_data, dict):
            for k, v in rules_data.items():
                left_terms = [s.strip() for s in str(k).split(',') if s.strip()]
                for left in left_terms:
                    pairs.append((left, v))

        elif isinstance(rules_data, list):
            for entry in rules_data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    pairs.append((entry[0], entry[1]))
                elif isinstance(entry, dict):
                    if 'if' in entry and 'then' in entry:
                        pairs.append((entry['if'], entry['then']))
                    elif 'from' in entry and 'to' in entry:
                        pairs.append((entry['from'], entry['to']))
                    elif 'input' in entry and 'output' in entry:
                        pairs.append((entry['input'], entry['output']))
                    else:
                        ks = list(entry.keys())
                        if len(ks) >= 2:
                            pairs.append((entry[ks[0]], entry[ks[1]]))

        return [(str(a), str(b)) for a, b in pairs]

    def find_x_intersect(p_left, p_right, level):
        x1, y1 = p_left
        x2, y2 = p_right
        if isclose(y2, y1):
            return None
        t = (level - y1) / (y2 - y1)
        if -1e-12 <= t <= 1 + 1e-12:
            return x1 + t * (x2 - x1)
        return None

    def clip_shape(pts, level):
        if not pts:
            return []
        alpha = float(level)
        tmp = []

        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]

            tmp.append((float(p1[0]), min(p1[1], alpha)))

            if (p1[1] - alpha) * (p2[1] - alpha) < -EPSILON:
                ix = find_x_intersect(p1, p2, alpha)
                if ix is not None:
                    tmp.append((float(ix), alpha))

        last_pt = pts[-1]
        tmp.append((float(last_pt[0]), min(last_pt[1], alpha)))

        tmp.sort(key=lambda p: p[0])

        uniq = []
        for x, y in tmp:
            if uniq and isclose(uniq[-1][0], x, abs_tol=EPSILON):
                uniq[-1] = (uniq[-1][0], max(uniq[-1][1], y))
            else:
                uniq.append((x, y))

        stitched = []
        idx = 0
        n = len(uniq)
        while idx < n:
            cx, cy = uniq[idx]
            if isclose(cy, alpha, abs_tol=EPSILON):
                j = idx
                while j + 1 < n and isclose(uniq[j + 1][1], alpha, abs_tol=EPSILON):
                    j += 1

                stitched.append((cx, alpha))
                if not isclose(uniq[j][0], cx, abs_tol=EPSILON):
                    stitched.append((uniq[j][0], alpha))
                idx = j + 1
            else:
                stitched.append((cx, cy))
                idx += 1

        result = []
        for p in stitched:
            if result and isclose(result[-1][0], p[0], abs_tol=1e-12) and isclose(result[-1][1], p[1], abs_tol=1e-12):
                continue
            result.append(p)
        return result

    in_sets = decode_lv(LVinput)
    out_sets = decode_lv(LVoutput)
    rule_pairs = normalize_rules(rules)

    mu_in = {term: get_membership(pts, T) for term, pts in in_sets.items()}

    log_msg("T =", float(T))
    log_msg("Степени входа =", json.dumps(mu_in, ensure_ascii=False))

    alpha_out = {name: 0.0 for name in out_sets}
    for in_term, out_term in rule_pairs:
        deg = mu_in.get(in_term, 0.0)
        if deg > alpha_out.get(out_term, 0.0):
            alpha_out[out_term] = float(deg)

    log_msg("Alpha уровни выходов =", json.dumps(alpha_out, ensure_ascii=False))

    clipped = {}
    for name, pts in out_sets.items():
        a = alpha_out.get(name, 0.0)
        if a <= EPSILON:
            clipped[name] = []
            log_msg(f"Clipping[{name}]: пропущен (0)")
        else:
            clipped_pts = clip_shape(pts, a)
            clipped[name] = clipped_pts
            dbg = [(round(x, 6), round(y, 6)) for x, y in clipped_pts]
            log_msg(f"Clipping[{name}]:", dbg)

    heights = [p[1] for shape in clipped.values() for p in shape]
    y_max = max(heights) if heights else 0.0
    log_msg("Global Max Y =", round(y_max, 6))

    if y_max <= EPSILON:
        centers = []
        for pts in out_sets.values():
            if pts:
                centers.append((pts[0][0] + pts[-1][0]) / 2.0)
        fallback = float(sum(centers) / len(centers)) if centers else 0.0
        log_msg("Fallback centroid =", round(fallback, 6))
        return fallback

    intervals = []
    for name, pts in clipped.items():
        if not pts:
            continue
        i = 0
        m = len(pts)
        while i < m:
            if isclose(pts[i][1], y_max, abs_tol=1e-7):
                j = i
                while j + 1 < m and isclose(pts[j + 1][1], y_max, abs_tol=1e-7):
                    j += 1
                l = float(pts[i][0])
                r = float(pts[j][0])
                intervals.append((l, r))
                i = j + 1
            else:
                i += 1

    log_msg("Raw max intervals =", [(round(l, 6), round(r, 6)) for l, r in intervals])

    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged = []
    for l, r in intervals_sorted:
        if not merged:
            merged.append([l, r])
        else:
            last = merged[-1]
            if l <= last[1] + EPSILON:
                last[1] = max(last[1], r)
            else:
                merged.append([l, r])

    final_intervals = [(float(a), float(b)) for a, b in merged]
    log_msg("Merged intervals =", [(round(a, 6), round(b, 6)) for a, b in final_intervals])

    L = sum(max(0.0, r - l) for l, r in final_intervals)
    log_msg("Total Length L =", round(L, 6))

    if L <= EPSILON:
        mids = [(l + r) / 2.0 for l, r in final_intervals]
        if not mids:
            xs = []
            for pts in clipped.values():
                for x, y in pts:
                    if isclose(y, y_max, abs_tol=1e-7):
                        xs.append(x)
            if not xs:
                return 0.0
            return float(sum(xs) / len(xs))

        res = sum(mids) / len(mids)
        log_msg("Degenerate centroid =", round(res, 6))
        return float(res)

    moment = 0.0
    for l, r in final_intervals:
        moment += (r * r - l * l)

    x_opt = 0.5 * moment / L
    log_msg("Moment sum =", round(moment, 6))
    log_msg("Result X_opt =", round(x_opt, 6))

    return float(x_opt)


if __name__ == '__main__':
    print(main('''
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
25, verbose=True))