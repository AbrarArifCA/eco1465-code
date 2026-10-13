"""Double-selection lasso and partialling out (Belloni, Chernozhukov and Hansen 2014).

The plain lasso penalises the treatment dummies too (at the 1-SE lambda it halves the
impacts effect). Double selection uses lasso only to choose controls:
  1. lasso of Y on the controls      -> S_Y
  2. lasso of each arm dummy on them -> S_D (union over the three arms)
  3. OLS of Y on D and S_Y U S_D, HC1 standard errors.
Partialling out regresses the lasso residual of Y on the lasso residuals of D.

lambda: the plug-in rule of hdm::rlasso (lambda = 2c sqrt(n) Phi^-1(1 - gamma/2p), c = 1.1,
gamma = 0.1/log n, heteroskedastic loadings from post-lasso residuals), and 5-fold CV with
the same seed for every lasso. In an experiment S_D should be (close to) empty.

Uses results of: ols, lasso
Output: table_double_lasso.tex
"""
import numpy as np
import statsmodels.api as sm
import sklearn.linear_model as skl
from scipy.stats import norm

import config as cfg
from Scripts import data
from Scripts.utils import save_table

TITLE = "Double-selection lasso and partialling out"


def center(Xm, v, w=None):
    """Centre X and v (weighted if w is given) and multiply rows by sqrt(w): an unweighted
    fit on the result is the weighted fit on the original data."""
    if w is None:
        return Xm - Xm.mean(0), v - v.mean()
    wn = w / w.mean()
    s = np.sqrt(wn)
    return ((Xm - np.average(Xm, axis=0, weights=wn)) * s[:, None],
            (v - np.average(v, weights=wn)) * s)


def rlasso(Xm, v, w=None, c=1.1, max_iter=15, tol=1e-8):
    """Plug-in (BCH) lasso with heteroskedastic penalty loadings, post-lasso residuals.
    hdm objective (1/n)||v - Xb||^2 + (lambda/n)||Psi b||_1  ->  sklearn alpha = lambda/(2n)
    after dividing column j by its loading psi_j."""
    Xc, vc = center(Xm, v, w)
    n, p = Xc.shape
    lam = 2 * c * np.sqrt(n) * norm.ppf(1 - (0.1 / np.log(n)) / (2 * p))
    e, psi = vc, None
    for _ in range(max_iter):
        psi_new = np.sqrt(((Xc ** 2) * (e ** 2)[:, None]).mean(0))
        if psi is not None and np.max(np.abs(psi_new - psi)) < tol:
            break
        psi = psi_new
        b = skl.Lasso(alpha=lam / (2 * n), fit_intercept=False, max_iter=10000).fit(Xc / psi, vc).coef_ / psi
        S = np.flatnonzero(b)
        e = vc - Xc[:, S] @ np.linalg.lstsq(Xc[:, S], vc, rcond=None)[0] if len(S) else vc.copy()
    return S, e


def cvlasso(Xm, v, w=None):
    """Lasso with lambda chosen by 5-fold CV on the shared seed; lasso residuals."""
    Xc, vc = center(Xm, v, w)
    Xs = Xc / Xc.std(0)
    lam_max = np.abs(Xs.T @ vc).max() / len(vc)
    grid = lam_max * 10 ** np.linspace(0, -4, 100)
    m = skl.ElasticNetCV(alphas=grid, l1_ratio=1, cv=data.kfold(), fit_intercept=False).fit(Xs, vc)
    return np.flatnonzero(m.coef_), vc - Xs @ m.coef_


def double_lasso(Xctrl, Dm, v, select, w=None):
    """Post-double selection and partialling out for a matrix of treatment dummies Dm."""
    SY, eY = select(Xctrl, v, w)
    SD, eD = zip(*[select(Xctrl, Dm[:, k], w) for k in range(Dm.shape[1])])
    U = sorted(set(SY).union(*SD))
    Z = sm.add_constant(np.column_stack([Dm, Xctrl[:, U]]))
    pds = (sm.WLS(v, Z, weights=w) if w is not None else sm.OLS(v, Z)).fit(cov_type="HC1")
    po = sm.OLS(eY, np.column_stack(eD)).fit(cov_type="HC1")     # rows already sqrt(w)-scaled
    k = Dm.shape[1]
    return {"SY": SY, "SD": SD, "U": U, "pds_b": pds.params[1:k + 1], "pds_se": pds.bse[1:k + 1],
                "po_b": po.params, "po_se": po.bse}


def report(ds, ctrl_names):
    for name, r in ds.items():
        print(f"--- {name} lambda: |S_Y| = {len(r['SY'])} of {len(ctrl_names)} controls; "
              f"|S_D| per arm = {[len(s) for s in r['SD']]}; union = {len(r['U'])}")
        for arm, s in zip(cfg.TREAT, r["SD"]):
            print(f"   S_D({arm}): {[ctrl_names[i] for i in s] or 'none'}")


def ds_table(ols_params, ols_bse, naive_coef, n_naive, ds, n_ctrl, fname):
    """Naive lasso vs OLS vs double selection vs partialling out."""
    def cell(b, se=None):
        return f"{b:.3f}" + (f" ({se:.3f})" if se is not None else "")
    rows = [("OLS, all controls", [cell(ols_params[t], ols_bse[t]) for t in cfg.TREAT], f"{n_ctrl}", "--"),
            (r"Lasso on $Y$ incl.\ $D$, 1-SE $\lambda$", [cell(b) for b in naive_coef], f"{n_naive}", "--")]
    for kind, bk, sk in [("Double selection", "pds_b", "pds_se"), ("Partialling out", "po_b", "po_se")]:
        for name in ["plug-in", "CV"]:
            r = ds[name]
            rows.append((rf"{kind}, {name} $\lambda$",
                         [cell(b, s) for b, s in zip(r[bk], r[sk])],
                         f"{len(r['SY'])}", "/".join(str(len(s)) for s in r["SD"])))
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r" & Climate impacts & Climate policies & Both & $|S_Y|$ & $|S_D|$ \\", r"\midrule"]
    for i, (lab, cells, sy, sd) in enumerate(rows):
        if i == 2:
            L.append(r"\addlinespace")
        L.append(" & ".join([lab] + cells + [sy, sd]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save_table("\n".join(L), fname, echo=True)


def run(ctx):
    md = data.main_design()
    X, Y = md.X, md.Y
    ols, lasso = ctx.need("ols"), ctx.need("lasso")

    Xctrl, Dm = X[md.CTRL].to_numpy(), X[cfg.TREAT].to_numpy()
    ds = {name: double_lasso(Xctrl, Dm, Y, f) for name, f in [("plug-in", rlasso), ("CV", cvlasso)]}
    report(ds, md.CTRL)

    naive = lasso["coef_1se_std"][:3] / md.X_scale[:3]       # first 3 columns are the arms
    ds_table(ols["params"], ols["bse"], naive, int((lasso["coef_1se_std"][3:] != 0).sum()),
             ds, len(md.CTRL), "table_double_lasso.tex")
    return {name: {k: (list(v) if k == "SD" else v) for k, v in r.items()} for name, r in ds.items()}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("double_lasso")
