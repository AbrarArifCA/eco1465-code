"""Paper's summary-statistics table and the support-by-arm figure (Project 1).

Outputs: table_sumstats.tex, fig_support_by_arm.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from Scripts import data
from Scripts.utils import save_table, save_fig, show_fig

TITLE = "Summary statistics table and support by arm"

ROWS = [
    ("Outcomes", [
        ("Supports carbon tax with cash transfers", "tax_transfers_bin"),
        ("Supports ban on combustion-engine cars", "standard_bin"),
        ("Supports green infrastructure program", "investments_bin"),
        ("Fairness of main policies index (z)", "index_fairness"),
        ("Willingness to change behavior index (z)", "index_willing_change")]),
    ("Treatment arms", [
        ("Control", "ctrl"), ("Climate impacts video", "Impacts"),
        ("Climate policies video", "Policies"), ("Both videos", "Both")]),
    ("Pre-treatment characteristics", [
        ("Female", "female"), ("College degree", "college01"), ("Age 50 or older", "age50"),
        ("Top income quartile", "incQ4"),
        ("Interest in politics ($-2$ to $+2$)", "interested_politics"),
        ("Left or very left (economic)", "left"), ("Right or very right (economic)", "right"),
        ("Urban", "urban"), ("Car dependent", "car_dependency"),
        ("Public transport availability ($-2$ to $+2$)", "availability_transport")]),
    ("Post-treatment beliefs (mediators)", [
        ("Believes own household loses (z)", "index_lose_policies_subjective"),
        ("Believes policies are effective (z)", "index_policies_effective")]),
    ("Real-stakes behavior", [
        ("Donates part of lottery prize", "donate_any"), ("Signs climate petition", "petition")]),
]

POLICIES = [("tax_transfers_support", "Carbon tax with cash transfers"),
            ("standard_support", "Ban on combustion-engine cars"),
            ("investments_support", "Green infrastructure program")]
ARMS = [("None", "Control", "0.45", "o"), ("Climate impacts", "Climate impacts video", "#1b9e77", "s"),
        ("Climate policy", "Climate policies video", "#d95f02", "D"),
        ("Both", "Both videos", "#7570b3", "^")]


def run(ctx):
    d = data.project_frame()
    c = d[d.arm == "None"]

    # ---- summary statistics table -------------------------------------------------------
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & Mean & SD & Min & Max & Control mean \\", r"\midrule"]
    for g, (head, vs) in enumerate(ROWS):
        if g:
            L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{head}}}}}\\")
        for lab, v in vs:
            x = d[v].astype(float)
            L.append(f"{lab} & {x.mean():.3f} & {x.std():.3f} & {x.min():.2f} & {x.max():.2f} "
                     f"& {c[v].astype(float).mean():.3f} \\\\")
    L += [r"\addlinespace", rf"Observations & \multicolumn{{4}}{{r}}{{{len(d):,}}} & {len(c):,} \\",
          r"\bottomrule", r"\end{tabular}"]
    save_table("\n".join(L), "table_sumstats.tex", echo=True)
    print("petition N =", d.petition.notna().sum())

    # ---- support by arm (dot plot, 95% CI) -----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    shares = {}
    for i, (v, lab) in enumerate(POLICIES):
        for j, (a, al, col, mk) in enumerate(ARMS):
            x = (d.loc[d.arm == a, v] > 0).astype(float)
            mu, se = x.mean(), x.std() / np.sqrt(len(x))
            shares[(lab, al)] = mu
            ax.errorbar(mu, i + (j - 1.5) * 0.17, xerr=1.96 * se, fmt=mk, color=col, ms=6,
                        capsize=2, label=al if i == 0 else None)
    ax.set_yticks(range(3))
    ax.set_yticklabels([l for _, l in POLICIES])
    ax.invert_yaxis()
    ax.set_xlabel("Share who somewhat or strongly support (95% CI)")
    ax.grid(axis="x", color="0.9")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig_support_by_arm.pdf")
    show_fig(fig)
    return {"support_shares": shares}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("descriptives")
