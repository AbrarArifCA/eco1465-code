"""Regression tree for Report .

Same main specification and the same 75 / 25 split as the nested-CV comparison, so test
MSEs are comparable. (1) a depth-3 tree to read; (2) the full tree, its cost-complexity
path, and alpha chosen by the inner 5-fold CV over a 100-point log grid; (3) test MSE.

Outputs: fig_tree_depth3.pdf, fig_tree_cv.pdf, fig_tree_pruned.pdf, fig_tree_pruned_readable.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from sklearn.tree import DecisionTreeRegressor as DTR, plot_tree, export_text

import config as cfg
from Scripts import data
from Scripts.utils import cv_plot, save_fig, show_fig, show, pretty

TITLE = "Regression tree: depth-3 tree, CV pruning, test MSE"


def run(ctx):
    md, sp = data.main_design(), data.split()
    names = md.feature_names
    pretty_names = [pretty(c) for c in names]
    X_train, X_test = md.Xa[sp.tr], md.Xa[sp.te]
    y_train, y_test = md.Y[sp.tr], md.Y[sp.te]

    # (1) depth-3 tree, unpruned, for reading
    reg = DTR(max_depth=3, random_state=cfg.SEED).fit(X_train, y_train)
    print(export_text(reg, feature_names=names, decimals=3))
    fig, ax = plt.subplots(figsize=(22, 8))
    anns = plot_tree(reg, feature_names=pretty_names, filled=True, rounded=True,
                     impurity=False, precision=3, fontsize=10, ax=ax)
    for t in anns:
        t.set_text(t.get_text().replace("value", "share supporting"))
    ax.set_title("Regression tree (depth 3), training sample: support for a carbon tax with cash "
                 "transfers\nleft branch = condition true, right branch = condition false", fontsize=12)
    save_fig(fig, "fig_tree_depth3.pdf")
    show_fig(fig)

    # (2) pruning by cross-validation
    full_tree = DTR(random_state=cfg.SEED).fit(X_train, y_train)
    a = full_tree.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
    print(f"full tree: {full_tree.tree_.n_leaves:,} leaves, depth {full_tree.get_depth()}, "
          f"{len(a):,} alphas on the pruning path")
    alphas_t = np.geomspace(a[a > 0].min(), a.max(), 100)
    G = skm.GridSearchCV(DTR(random_state=cfg.SEED), {"ccp_alpha": alphas_t}, refit=True,
                         cv=sp.inner, scoring="neg_mean_squared_error", n_jobs=-1).fit(X_train, y_train)
    best = G.best_estimator_
    print(f"CV-chosen alpha = {G.best_params_['ccp_alpha']:.3g}   leaves = {best.tree_.n_leaves}   "
          f"min CV MSE = {-G.best_score_:.5f}")

    tree_mse = np.column_stack([-G.cv_results_[f"split{k}_test_score"] for k in range(cfg.K)])
    show_fig(cv_plot(alphas_t, tree_mse, G.best_params_["ccp_alpha"],
                     "Regression tree: 5-fold CV error", "fig_tree_cv.pdf",
                     null_mse=y_train.var(), xlabel=r"$-\log(\alpha)$"))

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(best, feature_names=names, ax=ax)
    save_fig(fig, "fig_tree_pruned.pdf")
    show_fig(fig)

    fig, ax = plt.subplots(figsize=(50, 35))          # too large to show inline: saved only
    plot_tree(best, feature_names=pretty_names, filled=True, rounded=True, impurity=False,
              precision=1, fontsize=18, ax=ax)
    ax.set_title("Pruned regression tree based on selected Cross Validation: Support for a "
                 "carbon tax with cash transfers\n", fontsize=60)
    fig.tight_layout()
    save_fig(fig, "fig_tree_pruned_readable.pdf")
    show_fig(fig, keep=False)

    # (3) test MSE
    mse = lambda m: np.mean((y_test - m.predict(X_test)) ** 2)
    out = pd.Series({"Tree, unpruned": mse(full_tree), "Tree, depth 3": mse(reg),
                     "Tree, pruned (CV)": mse(best)}, name="test MSE")
    try:
        out = pd.concat([ctx.need("compare_linear", "test_mse"), out]).rename("test MSE")
    except RuntimeError:
        pass
    show(out.round(5))

    imp = pd.Series(best.feature_importances_, index=names, name="importance")
    print("treatment dummies:", imp[cfg.TREAT].round(4).to_dict())
    show(imp.sort_values(ascending=False).head(12).round(4))

    return {"test_mse": {"depth3": mse(reg), "pruned": mse(best), "unpruned": mse(full_tree)},
            "n_leaves": int(best.tree_.n_leaves), "ccp_alpha": G.best_params_["ccp_alpha"],
            "root_feature": names[reg.tree_.feature[0]], "importance": imp}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("regression_tree")
