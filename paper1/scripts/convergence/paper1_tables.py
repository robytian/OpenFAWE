"""Paper 1, Tables 7 and 10: time-step ladders and tree-versus-direct errors, recomputed from the evidence files.

Inputs (all in evidence/): convergence-c.json (cloud tree ladders), convergence-c-pc-rechecks.json
(direct-sum rechecks of the finest levels on the owner's PC), convergence-ll.json (LL + free wake
reference ladders), convergence-v3-a735.json (V3 second angle). Outputs: evidence/paper1-tables.json and
evidence/paper1-tables.md (the numbers to copy into the manuscript, nothing rounded in the JSON).

Definitions (Celik et al. 2008; verify before citing): levels are ordered coarse -> fine with ratio r = 2;
relative difference of neighbours (f_fine - f_coarse)/|f_fine|; for a monotone triple the observed order
p = ln|e21/e32| / ln r, Richardson value f3 + e32/(r^p - 1), GCI_fine = 1.25 |e32/f3| / (r^p - 1).
For a triple that changes sign or whose differences do not decrease, no order is computed; the script reports the largest neighbour difference of
the triple (absolute and relative to the fine value) as a plain range indicator. This is not the
least-squares procedure of Eca & Hoekstra (2014); the manuscript must not claim that it is.
Usage: paper1_tables.py
"""
import json
import math
from pathlib import Path

TASK = Path(__file__).resolve().parent
EVIDENCE = TASK/"evidence"
KEYS = ("CL", "CD", "CMy")
RATIO = 2.0


def triple(f1, f2, f3):
    e21, e32 = f2-f1, f3-f2
    oscillating = e21 == 0 or e32 == 0 or (e21 > 0) != (e32 > 0)
    if oscillating or abs(e32) >= abs(e21):
        # Sign change, or same sign with a growing difference: no observed order is defined.
        largest = max(abs(e21), abs(e32))
        return dict(monotone=False, kind="sign change" if oscillating else "differences not decreasing", e21=e21, e32=e32,
                    range_indicator_abs=largest, range_indicator_rel=largest/abs(f3))
    p = math.log(abs(e21/e32))/math.log(RATIO)
    return dict(monotone=True, e21=e21, e32=e32, observed_order=p, richardson=f3+e32/(RATIO**p-1),
                gci_fine=1.25*abs(e32/f3)/(RATIO**p-1))


def ladder(name, dt, levels, runs, source):
    out = dict(ladder=name, dt_s=dt, runs=runs, source=source, coefficients={})
    for key in KEYS:
        values = levels[key]
        out["coefficients"][key] = dict(
            values=values,
            neighbour_relative=[(values[i+1]-values[i])/abs(values[i+1]) for i in range(len(values)-1)],
            triples=[dict(levels=[i, i+1, i+2], **triple(*values[i:i+3])) for i in range(len(values)-2)])
    return out


def from_analysis(report, ladder_id, source):
    rows = report["ladders"][ladder_id]["rows"]
    levels = {key: [row[key] for row in rows] for key in KEYS}
    return ladder(ladder_id, [row["dt_s"] for row in rows], levels, [row["run"] for row in rows], source)


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    cloud = json.loads((EVIDENCE/"convergence-c.json").read_text(encoding="utf-8"))
    pc = json.loads((EVIDENCE/"convergence-c-pc-rechecks.json").read_text(encoding="utf-8"))
    table7 = []
    for model in ("V3", "M600"):
        entry = pc["dt_ladders"][model]
        table7.append(ladder(f"C1-{model}-dt (VSM + free wake; finest levels from the direct sum where run names start with c1d)",
                             entry["dt_s"], entry["levels"], entry["runs"], "evidence/convergence-c-pc-rechecks.json dt_ladders"))
        table7.append(from_analysis(cloud, f"C1-{model}-dt", "evidence/convergence-c.json (all levels tree theta 0.2)"))
    extra = []
    for name, ids in (("convergence-ll.json", ("LL-C1-V3-dt", "LL-C1-M600-dt")), ("convergence-v3-a735.json", ("C1-V3-a735-dt",))):
        path = EVIDENCE/name
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            extra.extend(from_analysis(report, i, f"evidence/{name}") for i in ids)
    # Table 10: tree against direct at the rechecked levels, with the level-to-level change of the direct ladder.
    table10 = []
    for check in pc["tree_vs_direct"]:
        model = check["model"]
        ladder_levels = pc["dt_ladders"][model]
        index = ladder_levels["runs"].index(check["direct_run"])
        row = dict(model=model, tree_run=check["tree_run"], direct_run=check["direct_run"], time_s=check["time_s"], coefficients={})
        for key in KEYS:
            tree, direct = check["values"][key]["tree"], check["values"][key]["direct"]
            level = ladder_levels["levels"][key]
            step = (level[index]-level[index-1])/abs(level[index])
            relative = (tree-direct)/abs(direct)
            row["coefficients"][key] = dict(tree=tree, direct=direct, tree_minus_direct_relative=relative,
                                            level_to_level_relative=step,
                                            ratio_tree_error_to_level_difference=abs(relative)/abs(step) if step else None,
                                            within_one_tenth=bool(step and abs(relative) <= 0.1*abs(step)))
        table10.append(row)
    result = dict(note="Recomputed by paper1_tables.py from the evidence files; see the module docstring for the definitions.",
                  table7=table7, reference_and_second_angle_ladders=extra, table10=table10)
    (EVIDENCE/"paper1-tables.json").write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    lines = ["# Paper 1 — Tables 7 and 10 recomputed from the evidence files (paper1_tables.py)", "",
             "Relative differences are (fine − coarse)/|fine|. p, Richardson and GCI (Fs 1.25) only for monotone triples;",
             "non-monotone triples carry the largest neighbour difference as a range indicator (not Eça–Hoekstra).", ""]
    for block in table7+extra:
        lines += [f"## {block['ladder']}", "", f"Source: `{block['source']}`; runs: {', '.join(block['runs'])}; Δt = {block['dt_s']} s", "",
                  "| | level 1 | level 2 | level 3 | level 4 | rel. 1→2 | rel. 2→3 | rel. 3→4 | triple 1–3 | triple 2–4 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for key in KEYS:
            c = block["coefficients"][key]
            cells = []
            for t in c["triples"]:
                cells.append(f"p {t['observed_order']:.2f}, GCI {100*t['gci_fine']:.4f}%, extrap. {t['richardson']:.6f}" if t["monotone"]
                             else f"{t['kind']}; range {t['range_indicator_abs']:.2e} ({100*t['range_indicator_rel']:.4f}%)")
            lines.append(f"| {key} | " + " | ".join(fmt(v) for v in c["values"]) + " | "
                         + " | ".join(f"{100*v:+.4f}%" for v in c["neighbour_relative"]) + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Table 10 — tree (θ 0.2, order 2, leaf 32) against the direct sum", "",
              "| run | coefficient | tree | direct | (tree − direct)/|direct| | level-to-level (direct ladder) | ratio | ≤ 1/10 |", "|---|---|---|---|---|---|---|---|"]
    for row in table10:
        for key in KEYS:
            c = row["coefficients"][key]
            lines.append(f"| {row['tree_run']} vs {row['direct_run']} | {key} | {c['tree']:.7f} | {c['direct']:.7f} | {c['tree_minus_direct_relative']:+.2e} |"
                         f" {c['level_to_level_relative']:+.2e} | {c['ratio_tree_error_to_level_difference']:.3f} | {'yes' if c['within_one_tenth'] else 'NO'} |")
    (EVIDENCE/"paper1-tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
