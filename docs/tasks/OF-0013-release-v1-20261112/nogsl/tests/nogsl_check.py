"""OF-0013 criterion N3: Makani's common/c_math run with GSL and with OpenFAWE's no-GSL subset on the same cases.

  nogsl_check.py gen CASES [--captured FILE]     write the cases (seeded; plus the captured controller matrices)
  nogsl_check.py check GSL_OUT NOGSL_OUT OUT.json

Cases: 10 000 random matrices of the controller's shapes 9x5, 9x6, 7x7, 8x8, 13x7 (2 000 each) with elements of
magnitude 1e-3..1e3 and random sign - 70 % full rank, 10 % a duplicated column, 5 % a zero column, 10 % a column that
is a combination of two others, 5 % a scaled duplicate - plus 500 matrices whose columns are permutations of one
another (equal column norms: pivot ties); each case in seven kinds (qr, qrpt, left, right, matleft k=3, rank at two
tolerances, svd). Bounds as OF-0013 section 3.3 N3; the singular values are read against the largest singular value
(the backward-error scale; a per-value reading is reported beside it).
"""
import json
import sys

import numpy as np

SHAPES = ((9, 5), (9, 6), (7, 7), (8, 8), (13, 7))
SEED = 20261112


def random_matrix(rng, m, n):
    mag = 10.0**rng.uniform(-3, 3, size=(m, n))
    return mag*rng.choice((-1.0, 1.0), size=(m, n))


def variant(rng, a):
    m, n = a.shape
    u = rng.uniform()
    if u < 0.70:
        return a, "full"
    i, j, l = rng.choice(n, size=3, replace=False)
    a = a.copy()
    if u < 0.80:
        a[:, j] = a[:, i]
        return a, "duplicate"
    if u < 0.85:
        a[:, j] = 0.0
        return a, "zero"
    if u < 0.95:
        a[:, j] = 0.7*a[:, i] - 1.3*a[:, l]
        return a, "combination"
    a[:, j] = 3.0*a[:, i]
    return a, "scaled"


def fmt(x):
    return " ".join(repr(float(v)) for v in np.ravel(x))


def gen(path, captured=None):
    rng = np.random.default_rng(SEED)
    lines, meta = [], []
    cid = 0

    def add(a, tag):
        nonlocal cid
        m, n = a.shape
        b_left = rng.uniform(-10, 10, size=m)
        b_right = rng.uniform(-10, 10, size=n)
        b_mat = rng.uniform(-10, 10, size=(m, 3))
        scale = float(np.abs(a).max()) if a.size else 1.0
        for kind, rhs, k, tol in (("qr", None, 0, 0.0), ("qrpt", None, 0, 0.0), ("left", b_left, 0, 0.0),
                                  ("right", b_right, 0, 0.0), ("matleft", b_mat, 3, 0.0),
                                  ("rank", None, 0, max(1e-9*scale, 2.3e-16)), ("rank", None, 0, 2.3e-16),
                                  ("svd", None, 0, 0.0)):
            lines.append(f"{kind} {cid} {m} {n} {k} {tol!r}\n{fmt(a)}" + (f"\n{fmt(rhs)}" if rhs is not None else "") + "\n")
            meta.append(dict(id=cid, kind=kind, tag=tag, m=m, n=n))
            cid += 1
    for m, n in SHAPES:
        for _ in range(2000):
            a, tag = variant(rng, random_matrix(rng, m, n))
            add(a, tag)
    for t in range(500):
        m, n = SHAPES[t % len(SHAPES)]
        col = random_matrix(rng, m, 1)[:, 0]
        a = np.stack([rng.permutation(col)*rng.choice((-1.0, 1.0), size=m) for _ in range(n)], axis=1)
        add(a, "tie")
    if captured:                    # the controller's own calls, each in its own kind with its own right-hand side
        seen = set()
        for line in open(captured):
            rec = json.loads(line)
            key = (rec["call"], tuple(rec["A"]), tuple(rec["rhs"]), rec["m"], rec["n"])
            if key in seen:
                continue
            seen.add(key)
            a = np.asarray(rec["A"], dtype=float).reshape(rec["m"], rec["n"])
            rhs = np.asarray(rec["rhs"], dtype=float) if rec["call"] != "qr" else None
            lines.append(f"{rec['call']} {cid} {rec['m']} {rec['n']} 0 0.0\n{fmt(a)}" + (f"\n{fmt(rhs)}" if rhs is not None else "") + "\n")
            meta.append(dict(id=cid, kind=rec["call"], tag="captured", m=rec["m"], n=rec["n"]))
            cid += 1
    with open(path, "w") as handle:
        handle.writelines(lines)
    with open(path + ".meta.json", "w") as handle:
        json.dump(meta, handle)
    print(json.dumps(dict(cases=len(lines), captured=sum(1 for m in meta if m["tag"] == "captured"))))


def parse(path):
    out = {}
    for line in open(path):
        w = line.split()
        kind, cid, i, rec = w[0], int(w[1]), 2, {}
        while i < len(w):
            key, n = w[i], int(w[i + 1])
            rec[key] = np.array([float(v) for v in w[i + 2:i + 2 + n]])
            i += 2 + n
        out[cid] = (kind, rec)
    return out


def check(gsl_path, nogsl_path, out_path):
    meta = {m["id"]: m for m in json.load(open(gsl_path.replace(".gsl.out", "") + ".meta.json"))} if False else None
    ga, na = parse(gsl_path), parse(nogsl_path)
    worst = {}
    fails = {}
    counts = {}

    def note(key, value, cid):
        if value > worst.get(key, (-1.0, None))[0]:
            worst[key] = (float(value), cid)

    def fail(key, cid):
        fails.setdefault(key, []).append(cid)

    for cid, (kind, g) in ga.items():
        kind2, o = na[cid]
        assert kind == kind2
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "qr":
            m = int(round(np.sqrt(len(g["Q"]))))
            n = len(g["R"])//m
            Q, R = o["Q"].reshape(m, m), o["R"].reshape(m, n)
            Rg = g["R"].reshape(m, n)
            A_scale = max(float(np.abs(Rg).max()), 1e-300)
            a_rec = Q@R
            a_ref = g["Q"].reshape(m, m)@Rg
            rec = float(np.abs(a_rec - a_ref).max())/A_scale
            note("qr_reconstruction", rec, cid)
            if rec > 1e-14:
                fail("qr_reconstruction", cid)
            if np.any(np.tril(R, -1) != 0.0):
                fail("qr_upper_triangular", cid)
            note("qr_R_vs_gsl", float(np.abs(R - Rg).max())/A_scale, cid)
        elif kind == "qrpt":
            if not np.array_equal(g["perm"], o["perm"]):
                fail("qrpt_permutation", cid)
            if not np.array_equal(g["status_signum"], o["status_signum"]):
                fail("qrpt_status_or_signum", cid)
            scale = max(float(np.abs(g["rdiag"]).max()), 1e-300)
            note("qrpt_rdiag_vs_gsl", float(np.abs(np.abs(o["rdiag"]) - np.abs(g["rdiag"])).max())/scale, cid)
        elif kind in ("left", "right", "matleft"):
            key = "X" if kind == "matleft" else "x"
            if g["ret"][0] != o["ret"][0]:
                fail(f"{kind}_error_code", cid)
            d = np.abs(o[key] - g[key])/(1.0 + np.abs(g[key]))
            v = float(d.max()) if d.size else 0.0
            note(f"{kind}_solution", v, cid)
            if v > 1e-12:
                fail(f"{kind}_solution", cid)
        elif kind == "rank":
            if g["rank"][0] != o["rank"][0]:
                fail("rank", cid)
        elif kind == "svd":
            if g["ret"][0] != o["ret"][0]:
                fail("svd_error_code", cid)
            s, sg = o["s"], g["s"]
            n = len(s)
            m = len(o["U"])//n
            smax = max(float(sg.max()), 1e-300)
            v = float(np.abs(s - sg).max())/smax
            note("svd_values_vs_largest", v, cid)
            if v > 1e-13:
                fail("svd_values", cid)
            big = sg >= 1e-8*smax
            if big.any():
                note("svd_values_per_value_(>=1e-8_smax)", float((np.abs(s - sg)[big]/sg[big]).max()), cid)
            if np.any(np.diff(s) > 0):
                fail("svd_order", cid)
            U, V = o["U"].reshape(m, n), o["V"].reshape(n, n)
            a_ref = g["U"].reshape(m, n)@np.diag(sg)@g["V"].reshape(n, n).T
            rec = float(np.abs(U@np.diag(s)@V.T - a_ref).max())/max(float(np.abs(a_ref).max()), 1e-300)
            note("svd_reconstruction", rec, cid)
            if rec > 1e-13:
                fail("svd_reconstruction", cid)
    try:
        tags = {m["id"]: m["tag"] for m in json.load(open(nogsl_path.rsplit("/", 1)[0] + "/cases.txt.meta.json"))}
    except OSError:
        tags = {}
    by_group = {}
    for key, ids in fails.items():
        for cid in ids:
            group = "captured" if tags.get(cid) == "captured" else "random"
            by_group.setdefault(group, {}).setdefault(key, 0)
            by_group[group][key] += 1
    result = dict(what="OF-0013 N3: Makani common/c_math with GSL vs with the OpenFAWE no-GSL subset",
                  failures_by_group=by_group,
                  cases=counts, worst={k: dict(value=v[0], case=v[1]) for k, v in worst.items()},
                  failures={k: dict(count=len(v), first=v[:10]) for k, v in fails.items()}, passed=not fails)
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--captured" else None)
    elif sys.argv[1] == "check":
        check(*sys.argv[2:5])
