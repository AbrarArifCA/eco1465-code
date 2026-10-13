"""Compliance with the video treatment (Project 1): share of each treated arm that reports
watching the whole video. In the Both arm, cells are climate video / policy video.

Output: table_compliance.tex
"""
from Scripts import data
from Scripts.utils import save_table

TITLE = "Compliance table (self-reported video watching)"

VIDEOS = {"Climate impacts": ["watched_climate"], "Climate policy": ["watched_policy"],
          "Both": ["watched_climate", "watched_policy"]}
CATS = [("Yes", "Watched the whole video"),
        ("No, there was a technical problem", "Did not watch: technical problem"),
        ("No, I skipped part of the video", "Did not watch: skipped part"),
        ("No answer", "No answer recorded")]


def run(ctx):
    d = data.project_frame()
    comp = {}
    for arm, vids in VIDEOS.items():
        s = d[d.arm == arm]
        comp[arm] = {v: s[v].astype(str).replace("", "No answer").value_counts(normalize=True)
                     for v in vids}

    L = [r"\begin{tabular}{lrrr}", r"\toprule",
         r" & Climate impacts & Climate policies & Both \\", r"\midrule"]
    for key, lab in CATS:
        cells = [" / ".join(f"{100 * comp[arm][v].get(key, 0):.1f}" for v in comp[arm])
                 for arm in VIDEOS]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\addlinespace")
    L.append("Respondents & " + " & ".join(f"{(d.arm == a).sum():,}" for a in VIDEOS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save_table("\n".join(L), "table_compliance.tex", echo=True)
    return {"compliance": {a: {v: s.to_dict() for v, s in r.items()} for a, r in comp.items()}}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("compliance")
