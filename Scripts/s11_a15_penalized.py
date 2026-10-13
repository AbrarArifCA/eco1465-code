"""Alternative specification: OLS, ridge, lasso and double selection on the paper's own
Table A15 specification.

"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.linear_model as skl
import sklearn.model_selection as skm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from Scripts import data
from Scripts.utils import save_table, show, path_plot, cv_plot, show_fig
from Scripts.s10_double_lasso import double_lasso, rlasso, cvlasso, report, ds_table

TITLE = "Table A15 specification: OLS, ridge, lasso, double selection (_a15 files)"

SHOW15 = {"const": "Intercept",
          "Impacts": "Climate impacts video", "Policies": "Climate policies video",
          "Both": "Both videos",
          "female": "Female", "other": "Gender: other", "children": "Has children",
          "age: 25-34": "Age 25--34", "age: 35-49": "Age 35--49", "age: 50+": "Age 50+",
          "income: Q2": "Income Q2", "income: Q3": "Income Q3", "income: Q4": "Income Q4",
          "educ: High-school non-College": "High school, no college",
          "educ: College degree": "College degree",
          "econ: Very left": "Very left (vs. left)", "econ: Center": "Centre (vs. left)",
          "econ: Right": "Right (vs. left)", "econ: Very right": "Very right (vs. left)",
          "econ: PNR": "Leaning: no answer"}


def run(ctx):
    warnings.filterwarnings("ignore")
    a = data.a15_design()
    X15, y15, w15 = a.X, a.y, a.w
    TREAT = cfg.TREAT
    print(f"X15: {X15.shape[0]:,} x {X15.shape[1]}   missing values: {int(X15.isna().sum().sum())}")

    # ---- OLS = Table A15 column (3) -----------------------------------------------------
    ols15 = sm.WLS(y15, sm.add_constant(X15), weights=w15).fit(cov_type="HC1")
    print(f"N = {int(ols15.nobs):,}   R2 = {ols15.rsquared:.3f}   (A15 col. 3: R2 = 0.101)")
    print("A15 col. 3: impacts 0.029 (0.008), policy 0.102 (0.008), both 0.130 (0.008)")
    show(pd.DataFrame({"coef": ols15.params[TREAT], "se": ols15.bse[TREAT]}).round(3))

    tab = ols15.summary2().tables[1].loc[list(SHOW15)].copy()
    tab.index = [SHOW15[k] for k in tab.index]
    tab.columns = ["Coef.", "Std. Err.", "$z$", "$P>|z|$", "[0.025", "0.975]"]
    body = tab.to_latex(float_format="%.3f", escape=False, column_format="lrrrrrr")
    body = body.replace("Climate impacts video &", r"\addlinespace" + "\n" + r"\multicolumn{7}{l}{\textit{Treatment arms (base: control)}}\\" + "\n" + "Climate impacts video &", 1)
    body = body.replace("Female &", r"\addlinespace" + "\n" + r"\multicolumn{7}{l}{\textit{Socioeconomic controls (paper's set A)}}\\" + "\n" + "Female &", 1)
    foot = (r"\addlinespace" + "\n" + rf"Observations & \multicolumn{{6}}{{r}}{{{int(ols15.nobs):,}}}\\" + "\n"
            + rf"$R^2$ & \multicolumn{{6}}{{r}}{{{ols15.rsquared:.3f}}}\\" + "\n"
            + r"Country fixed effects & \multicolumn{6}{r}{Yes (20 countries; base: US)}\\" + "\n" + r"\bottomrule")
    save_table(body.replace(r"\bottomrule", foot, 1), "table_ols_a15.tex")

    # ---- weighted ridge and lasso -------------------------------------------------------
    X15a, Y15 = np.asarray(X15), np.asarray(y15)
    wn   = w15 / w15.mean()                                   # sklearn normalises the same way
    sd15 = X15a.std(0)
    Xs15 = (X15a - np.average(X15a, axis=0, weights=wn)) / sd15
    Yc15 = Y15 - np.average(Y15, weights=wn)
    Xw, Yw = Xs15 * np.sqrt(wn)[:, None], Yc15 * np.sqrt(wn)  # weighted-least-squares transform
    null15 = np.average(Yc15 ** 2, weights=wn)
    kfold = data.kfold()

    lam_r15 = 10 ** np.linspace(6, -6, 100) / Y15.std()
    ridge_path15 = pd.DataFrame(skl.ElasticNet.path(Xw, Yw, l1_ratio=0., alphas=lam_r15)[1].T,
                                columns=X15.columns, index=-np.log(lam_r15))
    pipe_r15 = Pipeline([("scaler", StandardScaler()),
                         ("ridge", skl.ElasticNetCV(alphas=lam_r15, l1_ratio=0, cv=kfold))])
    pipe_r15.fit(X15a, Y15, ridge__sample_weight=w15)
    tr15 = pipe_r15.named_steps["ridge"]

    lam_max15 = np.abs(Xw.T @ Yw).max() / len(Yw)
    lam_l15 = lam_max15 * 10 ** np.linspace(0, -4, 100)
    lasso_path15 = pd.DataFrame(skl.Lasso.path(Xw, Yw, l1_ratio=1, alphas=lam_l15)[1].T,
                                columns=X15.columns, index=-np.log(lam_l15))
    pipe_l15 = Pipeline([("scaler", StandardScaler()),
                         ("lasso", skl.ElasticNetCV(alphas=lam_l15, l1_ratio=1, cv=kfold))])
    pipe_l15.fit(X15a, Y15, lasso__sample_weight=w15)
    tl15 = pipe_l15.named_steps["lasso"]

    m15, se15 = tl15.mse_path_.mean(1), tl15.mse_path_.std(1) / np.sqrt(cfg.K)
    lam_1se15 = tl15.alphas_[m15 <= m15.min() + se15[m15.argmin()]].max()
    lasso_1se15 = skl.Lasso(alpha=lam_1se15).fit(Xs15, Y15, sample_weight=w15)

    k = np.argmin(np.abs(lam_l15 - tl15.alpha_))
    direct = skl.Lasso(alpha=lam_l15[k]).fit(Xs15, Y15, sample_weight=w15).coef_
    print("check: max |path - direct fit| =", np.abs(lasso_path15.iloc[k].values - direct).max().round(6))
    print(f"ridge: CV lambda = {tr15.alpha_:.3g}")
    print(f"lasso: CV lambda = {tl15.alpha_:.3g}, non-zero {(tl15.coef_ != 0).sum()} of {X15.shape[1]}")
    print(f"lasso: 1-SE lambda = {lam_1se15:.3g}, non-zero {(lasso_1se15.coef_ != 0).sum()} of {X15.shape[1]}")
    print("dropped at 1-SE:", list(X15.columns[lasso_1se15.coef_ == 0]))

    show_fig(path_plot(ridge_path15, tr15.alpha_, "Ridge coefficient paths", "fig_ridge_path_a15.pdf"))
    show_fig(cv_plot(lam_r15, tr15.mse_path_, tr15.alpha_, "Ridge: 5-fold CV error",
                     "fig_ridge_cv_a15.pdf", null_mse=null15))
    show_fig(path_plot(lasso_path15, tl15.alpha_, "Lasso coefficient paths", "fig_lasso_path_a15.pdf"))
    show_fig(cv_plot(tl15.alphas_, tl15.mse_path_, tl15.alpha_, "Lasso: 5-fold CV error",
                     "fig_lasso_cv_a15.pdf", null_mse=null15, one_se=lam_1se15))

    # ---- comparison, entry order, weighted test MSE -------------------------------------
    comp15 = pd.DataFrame({"OLS (= A15)":  ols15.params[X15.columns].values,
                           "Ridge (CV)":   tr15.coef_ / sd15,
                           "Lasso (CV)":   tl15.coef_ / sd15,
                           "Lasso (1-SE)": lasso_1se15.coef_ / sd15}, index=X15.columns)
    show(comp15.loc[TREAT].round(4))

    lam_f, sol_f = skl.Lasso.path(Xw, Yw, l1_ratio=1, alphas=lam_max15 * 10 ** np.linspace(0, -4, 2000))[:2]
    first = pd.Series([np.argmax(r != 0) if (r != 0).any() else np.inf for r in sol_f], index=X15.columns)
    rank15 = first.rank(method="min").sort_values()
    print("entry rank of treatment dummies:", rank15[TREAT].astype(int).to_dict(), f"of {X15.shape[1]}")
    print("first to enter:", list(rank15.index[:6]))

    tr_i, te_i = next(skm.ShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE,
                                       random_state=cfg.SPLIT_SEED).split(Y15))
    inner = data.split().inner
    wmse = lambda pred: np.average((pred - Y15[te_i]) ** 2, weights=w15[te_i])
    fitw = lambda m, name: m.fit(X15a[tr_i], Y15[tr_i], **{f"{name}__sample_weight": w15[tr_i]})
    res15 = pd.Series({
        "Null":  wmse(np.average(Y15[tr_i], weights=w15[tr_i])),
        "OLS":   wmse(skl.LinearRegression().fit(X15a[tr_i], Y15[tr_i], sample_weight=w15[tr_i]).predict(X15a[te_i])),
        "Ridge": wmse(fitw(Pipeline([("s", StandardScaler()), ("m", skl.ElasticNetCV(alphas=lam_r15, l1_ratio=0, cv=inner))]), "m").predict(X15a[te_i])),
        "Lasso": wmse(fitw(Pipeline([("s", StandardScaler()), ("m", skl.ElasticNetCV(alphas=lam_l15, l1_ratio=1, cv=inner))]), "m").predict(X15a[te_i])),
    }, name="weighted test MSE")
    show(res15.round(4))

    # ---- double selection, weighted -----------------------------------------------------
    CTRL15 = [c for c in X15.columns if c not in TREAT]
    Xctrl15, Dm15 = X15[CTRL15].to_numpy(), X15[TREAT].to_numpy()
    ds15 = {name: double_lasso(Xctrl15, Dm15, Y15, f, w=w15) for name, f in [("plug-in", rlasso), ("CV", cvlasso)]}
    report(ds15, CTRL15)
    naive15 = lasso_1se15.coef_[:3] / sd15[:3]
    ds_table(ols15.params, ols15.bse, naive15, int((lasso_1se15.coef_[3:] != 0).sum()), ds15,
             len(CTRL15), "table_double_lasso_a15.tex")

    return {"ols_params": ols15.params, "ols_bse": ols15.bse, "comp": comp15, "test_mse": res15}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("a15_penalized")
