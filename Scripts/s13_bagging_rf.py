"""Bagging and random forest with an importance matrix (Report 4).

Both from RandomForestRegressor: bagging is max_features = p (all 53 regressors at every
split); a random forest uses max_features < p, chosen here by out-of-bag MSE on the training
data (the test set stays untouched). 500 trees each.

Outputs: fig_rf_importance.pdf, table_rf_importance.tex, rf_importance_full.csv
"""
import numpy as np  # noqa: I001
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.inspection import permutation_importance

import config as cfg
from Scripts import data
from Scripts.utils import save_fig, save_table, show_fig, show, pretty, table_path

TITLE = "Bagging and random forest; importance matrix"

BLOCKS = {"country": "Country (19 dummies)", "age": "Age (4)", "income": "Income (3)",
          "employment": "Employment (3)", "econ": "Economic leaning (5)"}


def block(c):
    for pre, lab in BLOCKS.items():
        if c.startswith(pre + ": "):
            return lab
    return "Treatment arms (3)" if c in cfg.TREAT else pretty(c)


def run(ctx):
    md, sp = data.main_design(), data.split()
    names = md.feature_names
    X_train, X_test = md.Xa[sp.tr], md.Xa[sp.te]
    y_train, y_test = md.Y[sp.tr], md.Y[sp.te]
    p = X_train.shape[1]
    forest = lambda mf: RF(max_features=mf, n_estimators=500, oob_score=True,
                           random_state=cfg.SEED, n_jobs=-1).fit(X_train, y_train)

    # bagging
    bag = forest(p)
    y_hat_bag = bag.predict(X_test)
    oob = {p: np.mean((y_train - bag.oob_prediction_) ** 2)}
    print(f"Bagging: OOB MSE = {oob[p]:.5f}   test MSE = {np.mean((y_test - y_hat_bag) ** 2):.5f}")
    del bag

    # random forest: max_features by OOB MSE
    for mf in [3, int(np.sqrt(p)), 13, p // 3, p // 2]:
        m = forest(mf)
        oob[mf] = np.mean((y_train - m.oob_prediction_) ** 2)
        del m
    oob = pd.Series(oob, name="OOB MSE").rename_axis("max_features").sort_index()
    show(oob.round(5))
    mf_best = int(oob.idxmin())
    rf = forest(mf_best)
    y_hat_rf = rf.predict(X_test)
    print(f"Random forest (max_features = {mf_best}): test MSE = {np.mean((y_test - y_hat_rf) ** 2):.5f}")

    # importance matrix
    perm = permutation_importance(rf, X_test, y_test, scoring="neg_mean_squared_error",
                                  n_repeats=5, random_state=cfg.SEED)
    fi = pd.DataFrame({"importance": rf.feature_importances_,
                       "permutation": perm.importances_mean,
                       "permutation_sd": perm.importances_std}, index=names)
    fi["rank_impurity"] = fi["importance"].rank(ascending=False).astype(int)
    fi["rank_permutation"] = fi["permutation"].rank(ascending=False).astype(int)
    fi = fi.sort_values(by="permutation", ascending=False)
    fi.to_csv(table_path("rf_importance_full.csv"))
    show(fi.head(20).round(4))
    del rf

    blocks = (fi.assign(block=[block(c) for c in fi.index])
              .groupby("block")[["importance", "permutation"]].sum()
              .sort_values("importance", ascending=False))
    show(blocks.round(4))

    # Generate figure for top 20 by permutation importance
    top = fi.head(20).iloc[::-1]
    colors = ["#d95f02" if c in cfg.TREAT else "#1f77b4" for c in top.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    axes[0].barh([pretty(c) for c in top.index], top["permutation"], xerr=top["permutation_sd"],
                 color=colors, ecolor="0.4")
    axes[0].set_xlabel("Increase in test MSE when permuted", fontsize=11)
    axes[0].set_title("Permutation importance (test set)", fontsize=12)
    axes[0].axvline(0, c="k", lw=0.6)
    axes[1].barh([pretty(c) for c in top.index], top["importance"], color=colors)
    axes[1].set_xlabel("Impurity importance (share of total)", fontsize=11)
    axes[1].set_title("Mean decrease in impurity (as in the lab)", fontsize=12)
    fig.suptitle(f"Random forest (500 trees, max_features = {mf_best}): 20 most important "
                 "regressors (treatment arms in orange)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig_rf_importance.pdf")
    show_fig(fig)

    # table: top 15 + any treatment arm not among them
    rows = list(fi.index[:15]) + [t for t in cfg.TREAT if t not in fi.index[:15]]
    tab = fi.loc[rows, ["permutation", "rank_permutation", "importance", "rank_impurity"]].copy()
    tab.index = [pretty(c) for c in tab.index]
    tab["permutation"] *= 1000
    tab.columns = [r"Permutation ($\times 10^{-3}$)", "Rank", "Impurity importance", "Rank"]
    body = tab.to_latex(float_format="%.3f", escape=False, column_format="lrrrr")
    if len(rows) > 15:
        body = body.replace(pretty(rows[15]) + " &", r"\addlinespace" + "\n" + pretty(rows[15]) + " &", 1)
    save_table(body, "table_rf_importance.tex")

    return {"test_mse": {"bagging": np.mean((y_test - y_hat_bag) ** 2),
                         "rf": np.mean((y_test - y_hat_rf) ** 2)},
            "mf_best": mf_best, "oob": oob, "importance": fi}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("bagging_rf")
