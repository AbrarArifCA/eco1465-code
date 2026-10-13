"""OLS vs. ridge vs. lasso (Report 3): coefficients on the OLS scale, and test MSE from
CV nested in a 75 / 25 split (lambda chosen by 5-fold CV inside the training sample).

Uses results of: ols, ridge, lasso
Output: table_ridge_lasso_coefficients.tex
"""
import warnings

import numpy as np
import pandas as pd
import sklearn.linear_model as skl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Scripts import data
from Scripts.utils import show, table_path, PRETTY, COUNTRY_NAMES

TITLE = "Compare OLS, ridge and lasso; nested-CV test MSE"


def clean_label(c):
    """Row names for the paper's coefficient table."""
    if c in PRETTY:
        return PRETTY[c]
    block, _, v = c.partition(": ")
    if block == "age":
        return f"Age {v.replace('-', '--')}"
    if block == "income":
        return f"Income {v}"
    if block == "employment":
        return f"Employment: {v}"
    if block == "econ":
        return f"Econ. leaning: {'no answer' if v == 'PNR' else v}"
    if block == "country":
        return f"Country: {COUNTRY_NAMES[v]}"
    return c


def run(ctx):
    warnings.filterwarnings("ignore")
    md = data.main_design()
    X, Xa, Y, X_scale = md.X, md.Xa, md.Y, md.X_scale
    ols, ridge, lasso = ctx.need("ols"), ctx.need("ridge"), ctx.need("lasso")

    # penalised coefficients are on the standardised scale: divide by sd(X) ("unstandardise")
    comp = pd.DataFrame({
        "OLS":          ols["params"][X.columns].values,
        "Ridge (CV)":   ridge["coef_std"] / X_scale,
        "Lasso (CV)":   lasso["coef_std"] / X_scale,
        "Lasso (1-SE)": lasso["coef_1se_std"] / X_scale,
    }, index=X.columns).round(4)
    show(comp)
    comp.to_latex(table_path("table_ridge_lasso_coefficients.tex"), float_format="%.4f",
                  escape=False, column_format="lrrrr")
    # same table with readable row names (the version used in the paper)
    comp.rename(index=clean_label).to_latex(table_path("table_ridge_lasso_coefficients_clean.tex"),
                                            float_format="%.4f", escape=False, column_format="lrrrr")
    print("wrote table_ridge_lasso_coefficients.tex and _clean.tex")

    # honest test error: CV nested within a validation split
    sp = data.split()
    tr, te = sp.tr, sp.te

    def test_mse(model):
        model.fit(Xa[tr], Y[tr])
        return np.mean((model.predict(Xa[te]) - Y[te]) ** 2)

    res = pd.Series({
        "Null (training mean)": np.mean((Y[tr].mean() - Y[te]) ** 2),
        "OLS":   test_mse(skl.LinearRegression()),
        "Ridge": test_mse(Pipeline([("scaler", StandardScaler()),
                                    ("m", skl.ElasticNetCV(alphas=ridge["lambdas"], l1_ratio=0, cv=sp.inner))])),
        "Lasso": test_mse(Pipeline([("scaler", StandardScaler()),
                                    ("m", skl.ElasticNetCV(alphas=lasso["lambdas"], l1_ratio=1, cv=sp.inner))])),
    }, name="test MSE")
    show(res.round(5))
    return {"coef_table": comp, "test_mse": res}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("compare_linear")
