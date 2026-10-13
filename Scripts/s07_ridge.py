"""Ridge regression (Report 3), following the ISLP Chapter 6 lab.

Drop the intercept, standardise X, fit the path over a log grid of lambda with
ElasticNet.path(l1_ratio=0), and choose lambda by 5-fold CV with ElasticNetCV inside a
Pipeline with StandardScaler. The lab's grid was tuned for Hitters (n = 263); with
n ~ 40,000 the grid is shifted down to 10**linspace(6, -6, 100) / sd(Y) so the plot
reaches the OLS end.

Outputs: fig_ridge_path.pdf, fig_ridge_cv.pdf
"""
import warnings

import numpy as np
import pandas as pd
import sklearn.linear_model as skl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Scripts import data
from Scripts.utils import path_plot, cv_plot, show_fig

TITLE = "Ridge: coefficient paths and 5-fold CV"


def run(ctx):
    warnings.filterwarnings("ignore")     # ElasticNet with l1_ratio=0 warns, as in the lab
    md = data.main_design()
    Xa, Y, Xs = md.Xa, md.Y, md.Xs

    lambdas = 10 ** np.linspace(6, -6, 100) / Y.std()
    soln = skl.ElasticNet.path(Xs, Y, l1_ratio=0., alphas=lambdas)[1]
    ridge_path = pd.DataFrame(soln.T, columns=md.X.columns, index=-np.log(lambdas))
    ridge_path.index.name = "negative log(lambda)"

    ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=data.kfold())
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", ridgeCV)]).fit(Xa, Y)
    tuned = pipe.named_steps["ridge"]
    m = tuned.mse_path_.mean(1)
    print(f"CV-chosen lambda = {tuned.alpha_:.3g}")
    print(f"min CV MSE       = {m.min():.5f}   null-model MSE = {Y.var():.5f}")
    print(f"CV MSE at smallest lambda (~OLS) minus minimum = {m[-1] - m.min():.2e}")

    show_fig(path_plot(ridge_path, tuned.alpha_, "Ridge coefficient paths", "fig_ridge_path.pdf"))
    show_fig(cv_plot(lambdas, tuned.mse_path_, tuned.alpha_, "Ridge: 5-fold CV error",
                     "fig_ridge_cv.pdf", null_mse=Y.var()))

    return {"lambdas": lambdas, "alpha": tuned.alpha_, "coef_std": tuned.coef_,
            "cv_mse_min": m.min()}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("ridge")
