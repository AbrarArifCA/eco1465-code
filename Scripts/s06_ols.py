"""OLS (linear probability model) on the main specification (Report 3).

Support for the carbon tax (0/1) on the three arm dummies, 16 scalar pre-treatment controls,
age / income / employment / economic-leaning dummies and 19 country dummies (53 columns);
HC1 standard errors; unweighted so OLS, ridge and lasso are comparable.

Output: table_ols.tex
"""
import statsmodels.api as sm

from Scripts import data
from Scripts.utils import save_table, show

TITLE = "OLS with all pre-treatment controls"

SHOW = {"const": "Intercept",
        "Impacts": "Climate impacts video", "Policies": "Climate policies video",
        "Both": "Both videos", "female": "Female", "college": "College degree",
        "urban": "Urban", "couple": "Lives as a couple", "children": "Has children",
        "hh_size": "Household size", "wealth": "Wealth quintile", "home_owner": "Home owner",
        "polluting_sector": "Works in polluting sector",
        "interested_politics": "Interest in politics",
        "econ: Left": "Left (vs. centre)", "econ: Very left": "Very left (vs. centre)",
        "car_dependency": "Car dependent", "availability_transport": "Public transport availability",
        "flights_agg": "Flights per year", "frequency_beef": "Beef consumption",
        "gas_expenses": "Gas expenses (per 100)"}


def run(ctx):
    md = data.main_design()
    X, y = md.X, md.y
    print(f"X: {X.shape[0]:,} x {X.shape[1]} (3 treatment dummies, {X.shape[1] - 3 - 19} "
          f"controls, 19 country dummies); mean of y = {y.mean():.3f}")

    ols = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1")
    print(f"N = {int(ols.nobs):,}   R2 = {ols.rsquared:.4f}")
    coef = ols.summary2().tables[1]
    show(coef.round(4).head(20))

    tab = coef.loc[list(SHOW)].copy()
    tab.loc["gas_expenses", ["Coef.", "Std.Err.", "[0.025", "0.975]"]] *= 100
    tab.index = [SHOW[k] for k in tab.index]
    tab.columns = ["Coef.", "Std. Err.", "$z$", "$P>|z|$", "[0.025", "0.975]"]
    body = tab.to_latex(float_format="%.3f", escape=False, column_format="lrrrrrr")
    body = body.replace("Climate impacts video &", r"\addlinespace" + "\n" + r"\multicolumn{7}{l}{\textit{Treatment arms (base: control)}}\\" + "\n" + "Climate impacts video &", 1)
    body = body.replace("Female &", r"\addlinespace" + "\n" + r"\multicolumn{7}{l}{\textit{Pre-treatment controls}}\\" + "\n" + "Female &", 1)
    foot = (r"\addlinespace" + "\n" + rf"Observations & \multicolumn{{6}}{{r}}{{{int(ols.nobs):,}}}\\" + "\n"
            + rf"$R^2$ & \multicolumn{{6}}{{r}}{{{ols.rsquared:.3f}}}\\" + "\n"
            + r"Country fixed effects & \multicolumn{6}{r}{Yes (19)}\\" + "\n" + r"\bottomrule")
    body = body.replace(r"\bottomrule", foot, 1)
    save_table(body, "table_ols.tex")

    return {"params": ols.params, "bse": ols.bse, "nobs": int(ols.nobs), "r2": ols.rsquared}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("ols")
