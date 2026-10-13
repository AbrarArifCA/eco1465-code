"""Boosting (Report 4): GradientBoostingRegressor with depth-3 trees.

1. 5,000 trees with learning rate 0.001 and 0.2, with training / test
   error curves (best iteration read off the test curve is for illustration only).
2. The reported model: learning rate 0.01, number of trees chosen by early stopping on 20% of
   the training data (stop after 100 trees without improvement), so the test set is unused.

Output: fig_boost_error.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as GBR

import config as cfg
from Scripts import data
from Scripts.utils import save_fig, show_fig, show

TITLE = "Boosting: learning-rate comparison and early stopping"


def run(ctx):
    md, sp = data.main_design(), data.split()
    X_train, X_test = md.Xa[sp.tr], md.Xa[sp.te]
    y_train, y_test = md.Y[sp.tr], md.Y[sp.te]
    staged = lambda m: np.array([np.mean((y_test - p) ** 2) for p in m.staged_predict(X_test)])

    fits = {}
    for lam in (0.001, 0.2):
        m = GBR(n_estimators=5000, learning_rate=lam, max_depth=3, random_state=cfg.SEED).fit(X_train, y_train)
        e = staged(m)
        fits[lam] = (m.train_score_, e)
        print(f"lambda = {lam}: test MSE after 5,000 trees = {e[-1]:.5f}   (lowest {e.min():.5f} "
              f"at tree {e.argmin() + 1:,}; read off the test set, for illustration only)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, lam in zip(axes, fits):
        train, test = fits[lam]
        idx = np.arange(train.shape[0])
        ax.plot(idx, train, "b", label="Training")
        ax.plot(idx, test, "r", label="Test")
        ax.set_title(rf"Boosting, $\lambda$ = {lam}", fontsize=12)
        ax.set_xlabel("Number of trees", fontsize=11)
        ax.legend()
    axes[0].set_ylabel("MSE", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "fig_boost_error.pdf")
    show_fig(fig)

    boost = GBR(n_estimators=5000, learning_rate=0.01, max_depth=3, validation_fraction=0.2,
                n_iter_no_change=100, random_state=cfg.SEED).fit(X_train, y_train)
    mse_es = np.mean((y_test - boost.predict(X_test)) ** 2)
    print(f"Boosting (lambda = 0.01, early stopping): {boost.n_estimators_:,} trees, test MSE = {mse_es:.5f}")
    imp = pd.Series(boost.feature_importances_, index=md.feature_names).sort_values(ascending=False)
    show(imp.head(10).round(4))

    return {"test_mse": {"lr0.001": fits[0.001][1][-1], "lr0.2": fits[0.2][1][-1], "early_stop": mse_es},
            "n_trees_early_stop": int(boost.n_estimators_), "importance": imp}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("boosting")
