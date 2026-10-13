"""Replication of the paper's Appendix Table A15: effect of each video on five outcomes,
WLS with survey weights, the paper's control set A and country FE, HC1 standard errors.

Output: table_a15.tex
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from Scripts import data
from Scripts.utils import save_table, show, stars

TITLE = "Replication of Table A15 (paper's specification)"

SET_A = ("female + other + children + C(age_control) + C(income_factor) + C(educ_categ) "
         "+ C(econ_leaning) + C(country)")
OUTS = [("investments_bin", r"\shortstack{Green\\infrastructure}"),
        ("standard_bin", r"\shortstack{Ban on\\combustion cars}"),
        ("tax_transfers_bin", r"\shortstack{Carbon tax with\\cash transfers}"),
        ("index_fairness", r"\shortstack{Fairness\\index}"),
        ("index_willing_change", r"\shortstack{Willingness to\\change behavior}")]
ARMS = [("Impacts", "Climate impacts video"), ("Policies", "Climate policies video"),
        ("Both", "Both videos")]


def run(ctx):
    d = data.project_frame()
    c = d[d.arm == "None"]
    fits = {v: smf.wls(f"{v} ~ Impacts + Policies + Both + {SET_A}", d,
                       weights=d.weight).fit(cov_type="HC1") for v, _ in OUTS}

    L = [r"\begin{tabular}{l" + "c" * len(OUTS) + "}", r"\toprule",
         " & " + " & ".join(h for _, h in OUTS) + r" \\",
         " & " + " & ".join(f"({i})" for i in range(1, len(OUTS) + 1)) + r" \\", r"\midrule"]
    for k, lab in ARMS:
        L.append(lab + " & " + " & ".join(
            f"${fits[v].params[k]:.3f}{stars(fits[v].pvalues[k])}$" for v, _ in OUTS) + r" \\")
        L.append(" & " + " & ".join(f"({fits[v].bse[k]:.3f})" for v, _ in OUTS) + r" \\")
    L.append(r"\addlinespace")
    L.append("Control group mean & " + " & ".join(
        f"{np.average(c[v], weights=c.weight):.3f}" for v, _ in OUTS) + r" \\")
    L.append("Observations & " + " & ".join(f"{int(fits[v].nobs):,}" for v, _ in OUTS) + r" \\")
    L.append("$R^2$ & " + " & ".join(f"{fits[v].rsquared:.3f}" for v, _ in OUTS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save_table("\n".join(L), "table_a15.tex", echo=True)

    coefs = pd.DataFrame({v: fits[v].params[[k for k, _ in ARMS]] for v, _ in OUTS}).T
    ses = pd.DataFrame({v: fits[v].bse[[k for k, _ in ARMS]] for v, _ in OUTS}).T
    show(coefs.round(3))
    return {"coef": coefs, "se": ses}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("table_a15")
