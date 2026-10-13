"""Test-error comparison of every predictive model (Report 4), on the same 25% test set.

Uses results of: compare_linear, regression_tree, bagging_rf, boosting
Output: table_error_comparison.tex
"""
import numpy as np
import pandas as pd

from Scripts import data
from Scripts.utils import save_table, show

TITLE = "Test MSE / RMSE comparison table"

# The Report 4 / Project 1 table leaves out ridge and lasso (their test MSE equals OLS's);
# set to True to add the two rows back.
INCLUDE_RIDGE_LASSO = False


def run(ctx):
    md, sp = data.main_design(), data.split()
    y_test = md.Y[sp.te]
    lin = ctx.need("compare_linear", "test_mse")
    tree = ctx.need("regression_tree", "test_mse")
    rf = ctx.need("bagging_rf")
    bst = ctx.need("boosting")

    errors = pd.Series({
        "Null (training mean)":                       lin["Null (training mean)"],
        "OLS":                                        lin["OLS"],
        "Ridge (CV)":                                 lin["Ridge"],
        "Lasso (CV)":                                 lin["Lasso"],
        "Regression tree, depth 3":                   tree["depth3"],
        "Regression tree, pruned by CV":              tree["pruned"],
        "Bagging (500 trees)":                        rf["test_mse"]["bagging"],
        f"Random forest (500 trees, $m$ = {rf['mf_best']})": rf["test_mse"]["rf"],
        "Boosting, $\\lambda$ = 0.001 (5,000 trees)": bst["test_mse"]["lr0.001"],
        "Boosting, $\\lambda$ = 0.2 (5,000 trees)":   bst["test_mse"]["lr0.2"],
        f"Boosting, $\\lambda$ = 0.01 (early stop, {bst['n_trees_early_stop']:,} trees)": bst["test_mse"]["early_stop"],
    }, name="Test MSE")
    if not INCLUDE_RIDGE_LASSO:
        errors = errors.drop(["Ridge (CV)", "Lasso (CV)"])

    tab = pd.DataFrame({"Test MSE": errors, "Test RMSE": np.sqrt(errors)})
    tab["vs. OLS (%)"] = 100 * (tab["Test MSE"] / errors["OLS"] - 1)
    tab["$R^2$ (test)"] = 1 - tab["Test MSE"] / np.var(y_test)
    show(tab.round(4))

    body = tab.rename(columns={"vs. OLS (%)": r"vs. OLS (\%)"}).to_latex(
        escape=False, column_format="lrrrr",
        formatters={"Test MSE": "{:.4f}".format, "Test RMSE": "{:.4f}".format,
                    r"vs. OLS (\%)": lambda v: f"{v:+.1f}",
                    "$R^2$ (test)": lambda v: f"{0.0 if abs(v) < 5e-5 else v:.3f}"})
    body = body.replace("+0.0 ", "0.0 ").replace("+-", "-")
    for first in ["OLS &", "Regression tree, depth 3 &", "Bagging (500 trees) &"]:
        body = body.replace(first, r"\addlinespace" + "\n" + first, 1)
    save_table(body, "table_error_comparison.tex", echo=True)
    return {"errors": tab}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("error_comparison")
