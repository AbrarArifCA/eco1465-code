"""Lasso (Report 3): CV-chosen and 1-SE lambda, coefficient paths, order of entry.

The grid runs from lambda_max (all coefficients zero) down four orders of magnitude, as
ElasticNetCV does internally; building it by hand keeps the code working across sklearn
versions. Lasso.path needs l1_ratio=1 (its default is 0.5).

Outputs: fig_lasso_path.pdf, fig_lasso_cv.pdf
"""
import warnings

import numpy as np
import pandas as pd
import sklearn.linear_model as skl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from Scripts import data
from Scripts.utils import path_plot, cv_plot, show_fig, show

TITLE = "Lasso: CV and 1-SE lambda, paths, order of entry"


def run(ctx):
    warnings.filterwarnings("ignore")
    md = data.main_design()
    X, Xa, Y, Xs = md.X, md.Xa, md.Y, md.Xs

    lam_max = np.abs(Xs.T @ (Y - Y.mean())).max() / len(Y)
    lambdas = lam_max * 10 ** np.linspace(0, -4, 100)

    lassoCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=1, cv=data.kfold())
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("lasso", lassoCV)]).fit(Xa, Y)
    tuned = pipe.named_steps["lasso"]

    m = tuned.mse_path_.mean(1)
    se = tuned.mse_path_.std(1) / np.sqrt(cfg.K)
    i_min = m.argmin()
    lam_1se = tuned.alphas_[m <= m[i_min] + se[i_min]].max()
    lasso_1se = skl.Lasso(alpha=lam_1se).fit(Xs, Y)
    print(f"CV-chosen lambda = {tuned.alpha_:.3g}   non-zero: {(tuned.coef_ != 0).sum()} of {X.shape[1]}")
    print(f"1-SE lambda      = {lam_1se:.3g}   non-zero: {(lasso_1se.coef_ != 0).sum()} of {X.shape[1]}")
    print(f"min CV MSE       = {m.min():.5f}")
    print("dropped at CV min:", list(X.columns[tuned.coef_ == 0]))
    print("dropped at 1-SE:  ", list(X.columns[lasso_1se.coef_ == 0]))

    lambdas_p, soln = skl.Lasso.path(Xs, Y, l1_ratio=1, alphas=lambdas)[:2]
    lasso_path = pd.DataFrame(soln.T, columns=X.columns, index=-np.log(lambdas_p))
    lasso_path.index.name = "negative log(lambda)"
    k = np.argmin(np.abs(lambdas_p - tuned.alpha_))
    print("check: max |path - direct fit| =",
          np.abs(soln[:, k] - skl.Lasso(alpha=lambdas_p[k]).fit(Xs, Y).coef_).max().round(6))

    show_fig(path_plot(lasso_path, tuned.alpha_, "Lasso coefficient paths", "fig_lasso_path.pdf"))
    show_fig(cv_plot(tuned.alphas_, tuned.mse_path_, tuned.alpha_, "Lasso: 5-fold CV error",
                     "fig_lasso_cv.pdf", null_mse=Y.var(), one_se=lam_1se))

    # order of entry on a finer grid (largest lambda at which each variable becomes non-zero)
    lam_fine, sol_fine = skl.Lasso.path(Xs, Y, l1_ratio=1,
                                        alphas=lam_max * 10 ** np.linspace(0, -4, 2000))[:2]
    fine = pd.DataFrame(sol_fine.T, columns=X.columns, index=-np.log(lam_fine))
    nz = fine.ne(0)
    entry = pd.Series({c: (-fine.index[nz[c].values][0] if nz[c].any() else np.nan)
                       for c in X.columns}, name="log(lambda) at entry").sort_values(ascending=False)
    print("First variables to enter the lasso path:")
    show(entry.head(12))

    return {"lambdas": lambdas, "alpha": tuned.alpha_, "coef_std": tuned.coef_,
            "lam_1se": lam_1se, "coef_1se_std": lasso_1se.coef_, "entry": entry,
            "cv_mse_min": m.min()}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("lasso")
