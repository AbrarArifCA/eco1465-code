"""Covariate balance across the four arms (Project 1).

Each row: control mean and the difference from control for each arm (regression on the arm
dummy with country FE, HC1). Bottom rows: joint F-test that the covariates do not predict
assignment. The figure shows standardized differences.

Outputs: table_balance.tex, fig_balance.pdf
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from Scripts import data
from Scripts.utils import save_table, save_fig, show_fig, show, stars

TITLE = "Balance table and standardized-difference plot"

BAL_VARS = [("Female", "female"), ("College degree", "college01"), ("Age 50 or older", "age50"),
            ("Top income quartile", "incQ4"), ("Has children", "children"),
            ("Interest in politics", "interested_politics"), ("Left or very left", "left"),
            ("Right or very right", "right"), ("Urban", "urban"), ("Car dependent", "car_dependency"),
            ("Public transport availability", "availability_transport")]
ARMS = [("Impacts", "Climate impacts", "Climate impacts"),
        ("Policies", "Climate policy", "Climate policies"),
        ("Both", "Both", "Both")]


def run(ctx):
    db = data.project_frame().dropna(subset=["urban"])

    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & Control & \multicolumn{3}{c}{Difference from control} \\", r"\cmidrule(lr){3-5}",
         r" & mean & Climate impacts & Climate policies & Both \\", r"\midrule"]
    smd = {}
    for lab, v in BAL_VARS:
        cells = [f"{db.loc[db.arm == 'None', v].mean():.3f}"]
        for k, arm, name in ARMS:
            s = db[db.arm.isin(["None", arm])]
            m = smf.ols(f"{v} ~ {k} + C(country)", s).fit(cov_type="HC1")
            cells.append(f"${m.params[k]:.3f}{stars(m.pvalues[k])}$")
            sd = s[v].std()
            smd.setdefault(lab, {})[name] = (s.loc[s[k] == 1, v].mean() - s.loc[s[k] == 0, v].mean()) / sd
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\addlinespace")

    joint = []
    f = " + ".join(v for _, v in BAL_VARS)
    for k, arm, _ in ARMS:
        s = db[db.arm.isin(["None", arm])]
        m = smf.ols(f"{k} ~ {f} + C(country)", s).fit(cov_type="HC1")
        ft = m.f_test(", ".join(f"{v} = 0" for _, v in BAL_VARS))
        joint.append((float(ft.fvalue), float(ft.pvalue), int(m.nobs)))
    L.append("Joint $F$-test, $p$-value & & " + " & ".join(f"{p:.3f}" for _, p, _ in joint) + r" \\")
    L.append("Observations (arm + control) & & " + " & ".join(f"{n:,}" for _, _, n in joint) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save_table("\n".join(L), "table_balance.tex", echo=True)
    joint = pd.DataFrame(joint, columns=["F", "p", "N"], index=[a[2] for a in ARMS])
    show(joint.round(3))

    smd = pd.DataFrame(smd).T
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    yy = np.arange(len(smd))
    for j, (arm, mk, col) in enumerate([("Climate impacts", "o", "#1b9e77"),
                                        ("Climate policies", "s", "#d95f02"),
                                        ("Both", "^", "#7570b3")]):
        ax.scatter(smd[arm], yy + (j - 1) * 0.18, marker=mk, color=col, label=arm, zorder=3)
    ax.axvline(0, c="k", lw=0.8)
    for x in (-0.05, 0.05):
        ax.axvline(x, c="0.6", ls="--", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(smd.index)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized difference from control (treated - control) / SD")
    ax.set_xlim(-0.1, 0.1)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig_balance.pdf")
    show_fig(fig)
    return {"joint": joint, "smd": smd}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("balance")
