"""Summary statistics (Reports 1-2): descriptives, outcomes by arm, the paper's Table A15
estimates, balance, and data-quality checks.

Outputs: summary_table_key_variables.tex/.csv, table_descriptives.tex/.csv, table_a15_country_fe.tex

Findings. 40,680 respondents in 20 countries. Support is majoritarian for green
infrastructure (68%) but split for the ban on combustion-engine cars (55%) and the carbon
tax with cash transfers (53%). Arm shares match the design (29.0% control, ~23.6% each
treated arm), and the treatment-effect estimates reproduce Appendix Table A15. Caveats: the file is the
post-screening sample (attrition in Table A32 cannot be assessed), the combined arm
marginally fails a joint balance test, about 14% of treated respondents do not confirm
watching their video, `college` is coded 1/2, three respondents are coded "Below 18", and
`duration` has an 8-day maximum.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import config as cfg
from Scripts import data
from Scripts.utils import show, save_table, table_path

TITLE = "Summary statistics, Table A15 estimates, balance and data quality"

OUTCOMES = ["standard_support", "investments_support", "tax_transfers_support",
            "index_fairness", "index_willing_change", "index_main_policies"]
QUALITY  = ["duration", "duration_wo_treatment", "donation_fraction"]
RAW_COV  = ["female", "college", "income_factor", "urban",
            "car_dependency", "home_owner", "econ_leaning", "age"]
ADMIN    = [cfg.ARM, cfg.COUNTRY, cfg.WEIGHT, "treatment_climate", "treatment_policy",
            "valid", "rush", "rush_treatment", "finished_attentive",
            "attention_test", "watched_climate", "watched_policy", "petition"]
BIN = ["standard_support_bin", "investments_support_bin", "tax_transfers_support_bin"]
COVARIATES = ["female", "college01", "income_factor", "urban",
              "car_dependency", "home_owner", "econ_leaning"]

# (category, label, variable, scale) for the key-variable table
SPEC = [
    ("Treatment identifiers", "Control group",                  "ctrl",                    "Binary {0,1}"),
    ("Treatment identifiers", "Climate impacts treatment",       "t_impacts",               "Binary {0,1}"),
    ("Treatment identifiers", "Climate policies treatment",      "t_policies",              "Binary {0,1}"),
    ("Treatment identifiers", "Both treatments combined",        "t_both",                  "Binary {0,1}"),
    ("Primary stated outcomes", "Green infrastructure support",  "investments_support_bin", "Binary {0,1}"),
    ("Primary stated outcomes", "Ban on combustion-engine cars", "standard_support_bin",    "Binary {0,1}"),
    ("Primary stated outcomes", "Carbon tax w/ cash transfers",  "tax_transfers_support_bin", "Binary {0,1}"),
    ("Primary stated outcomes", "Support main policies index",   "index_main_policies",     "Std. z-score"),
    ("Primary stated outcomes", "Fairness of main policies index", "index_fairness",        "Std. z-score"),
    ("Real-stakes behaviours", "Willing to donate lottery prize", "willing_donate",         "Binary {0,1}"),
    ("Real-stakes behaviours", "Share of prize donated",          "donation_fraction",      "Continuous [0,1]"),
    ("Real-stakes behaviours", "Willing to sign climate petition", "petition",              "Binary {0,1}"),
    ("First-stage belief indices", "Perceived policy effectiveness", "index_policies_effective", "Std. z-score"),
    ("First-stage belief indices", "Believes own household loses",   "index_lose_policies_subjective", "Std. z-score"),
    ("First-stage belief indices", "Believes low-income earners lose", "index_lose_policies_poor", "Std. z-score"),
    ("First-stage belief indices", "Knowledge index",                 "index_knowledge",    "Std. z-score"),
]
EXTRA = ["index_policies_effective", "index_lose_policies_subjective",
         "index_lose_policies_poor", "index_knowledge"]

def w_mean(x, w):
    m = x.notna() & w.notna()
    return np.average(x[m], weights=w[m]) if m.sum() else np.nan

def w_std(x, w):
    m = x.notna() & w.notna()
    if m.sum() < 2:
        return np.nan
    mu = np.average(x[m], weights=w[m])
    var = np.average((x[m] - mu) ** 2, weights=w[m])
    return np.sqrt(var * m.sum() / (m.sum() - 1))

def describe(df, cols, weights=cfg.WEIGHT):
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        r = {"variable": c, "N": int(x.notna().sum()), "miss_pct": 100 * x.isna().mean(),
                 "mean": x.mean(), "median": x.median(), "sd": x.std(), "min": x.min(),
                 "p25": x.quantile(.25), "p75": x.quantile(.75), "max": x.max(),
                 "unique": int(x.nunique())}
        w = pd.to_numeric(df[weights], errors="coerce")
        r["w_mean"], r["w_sd"] = w_mean(x, w), w_std(x, w)
        rows.append(r)
    return pd.DataFrame(rows).set_index("variable").round(3)

def by_arm(df, cols, ctrl=1):
    """Mean in each arm, difference vs control, Welch p-value."""
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        base = x[df[cfg.ARM] == ctrl].dropna()
        r = {"variable": c, "control": base.mean()}
        for lv in sorted(df[cfg.ARM].dropna().unique()):
            if lv == ctrl:
                continue
            t = x[df[cfg.ARM] == lv].dropna()
            r[cfg.ARM_LABELS[lv]] = t.mean()
            r[f"diff_{lv}"] = t.mean() - base.mean()
            r[f"p_{lv}"] = stats.ttest_ind(t, base, equal_var=False)[1]
        rows.append(r)
    return pd.DataFrame(rows).set_index("variable").round(4)

def table_a15(df, outcomes):
    """Paper's specification: arm dummies + country FE, survey weights, HC1 (Table A15)."""
    rows = []
    for y in outcomes:
        d = df.dropna(subset=[y]).copy()
        d["_y"] = pd.to_numeric(d[y], errors="coerce")
        m = smf.wls(f"_y ~ C({cfg.ARM}) + C({cfg.COUNTRY})", data=d,
                    weights=d[cfg.WEIGHT]).fit(cov_type="HC1")
        r = {"outcome": y, "N": int(m.nobs),
             "control_mean": round(d.loc[d[cfg.ARM] == 1, "_y"].mean(), 3)}
        for lv in (2, 3, 4):
            k = f"C({cfg.ARM})[T.{lv}]"
            r[cfg.ARM_LABELS[lv]] = round(m.params[k], 4)
            r[f"se_{lv}"] = round(m.bse[k], 4)
        rows.append(r)
    return pd.DataFrame(rows).set_index("outcome")

def balance(df, covariates, ctrl=1):
    """Treatment dummy on all covariates + country FE; joint F-test of the covariates."""
    out = []
    for lv in sorted(df[cfg.ARM].dropna().unique()):
        if lv == ctrl:
            continue
        s = df[df[cfg.ARM].isin([ctrl, lv])].copy()
        s["_T"] = (s[cfg.ARM] == lv).astype(int)
        m = smf.ols(f"_T ~ {' + '.join(covariates)} + C({cfg.COUNTRY})", data=s).fit(cov_type="HC1")
        terms = [t for t in m.params.index if any(t.startswith(c) for c in covariates)]
        f = m.f_test(" = 0, ".join(terms) + " = 0")
        out.append({
            "arm": cfg.ARM_LABELS[lv],
            "N": int(m.nobs),
            "R2": round(m.rsquared, 5),
            "F": round(float(f.fvalue), 3),
            "p": round(float(f.pvalue), 4)
        })
    return pd.DataFrame(out)

def tex_num(v, dp=3):
    if pd.isna(v):
        return ""
    if abs(v) < 0.5 * 10 ** (-dp):
        v = 0.0
    return f"{v:,.{dp}f}".replace(",", "{,}")

def key_variable_table(summary):
    scale = {"Binary {0,1}": "Binary", "Continuous [0,1]": r"Cont.\ $[0,1]$",
             "Std. z-score": r"Std.\ $z$"}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Summary statistics for key variables}", r"\label{tab:keyvars}",
         "% If your document is 12pt, change \\footnotesize to \\scriptsize.",
         r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\renewcommand{\arraystretch}{1.08}",
         r"\begin{tabular}{@{}ll rrrrrrrr@{}}", r"\toprule",
         r"Variable & Scale & $N$ & Miss.\% & Mean & Median & SD & Min & Max & Wtd.\ mean \\",
         r"\midrule"]
    for i, cat in enumerate(summary["Category"].unique()):
        if i:
            L.append(r"\addlinespace[3pt]")
        L.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{{cat}}}}}\\[1pt]")
        for _, r in summary[summary["Category"] == cat].iterrows():
            L.append(" & ".join([r"\quad " + r["Variable"], scale[r["Type / scale"]],
                                 tex_num(r["N"], 0), tex_num(r["Miss %"], 2),
                                 tex_num(r["Mean"]), tex_num(r["Median"]), tex_num(r["SD"]),
                                 tex_num(r["Min"]), tex_num(r["Max"]),
                                 tex_num(r["Wtd mean"])]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)

def run(ctx):
    df = data.read(ADMIN + OUTCOMES + QUALITY + RAW_COV + EXTRA).copy()
    print(f"{df.shape[0]:,} rows")

    # ---- derived variables -----------------------------------------------------------
    print("college raw coding (1/2, not 0/1):", df["college"].value_counts().to_dict())
    for c in ["standard_support", "investments_support", "tax_transfers_support"]:
        df[c + "_bin"] = (pd.to_numeric(df[c], errors="coerce") > 0).astype(float)
    df["college01"] = (df["college"] == 1).astype(float)
    df["ctrl"]       = (df[cfg.ARM] == 1).astype(float)
    df["t_impacts"]  = (df[cfg.ARM] == 2).astype(float)
    df["t_policies"] = (df[cfg.ARM] == 3).astype(float)
    df["t_both"]     = (df[cfg.ARM] == 4).astype(float)
    df["willing_donate"] = (pd.to_numeric(df["donation_fraction"], errors="coerce") > 0).astype(float)

    # ---- 1. descriptive statistics ------------------------------------------------------
    print("\n1. Descriptive statistics (unweighted and weighted)")
    desc = describe(df, OUTCOMES + BIN + COVARIATES + QUALITY)
    show(desc)

    # ---- 2. key-variable table ----------------------------------------------------------
    rows = []
    for cat, label, var, scale in SPEC:
        x = pd.to_numeric(df[var], errors="coerce")
        w = pd.to_numeric(df[cfg.WEIGHT], errors="coerce")
        rows.append({"Category": cat, "Variable": label, "Type / scale": scale,
                     "N": int(x.notna().sum()), "Miss %": round(100 * x.isna().mean(), 2),
                     "Mean": x.mean(), "Median": x.median(), "SD": x.std(),
                     "Min": x.min(), "Max": x.max(), "Wtd mean": w_mean(x, w)})
    summary = pd.DataFrame(rows)
    print("\n2. Key variables")
    show(summary.round(3))
    summary.to_csv(table_path("summary_table_key_variables.csv"), index=False)
    save_table(key_variable_table(summary), "summary_table_key_variables.tex")

    # ---- 3. randomisation ---------------------------------------------------------------
    counts = df[cfg.ARM].value_counts().sort_index()
    print("\n3. Arm shares vs. design")
    show(pd.DataFrame({"arm": [cfg.ARM_LABELS[i] for i in counts.index], "n": counts.values,
                       "share": (counts / counts.sum()).round(4).values,
                       "paper": [0.294, 0.235, 0.235, 0.235]}))
    show(pd.crosstab([df.treatment_climate, df.treatment_policy], df[cfg.ARM]))

    # ---- 4. outcomes by arm and Table A15 -----------------------------------------------
    print("\n4. Outcomes by arm (unconditional)")
    show(by_arm(df, BIN + ["index_fairness", "index_willing_change"]))
    a15_tab = table_a15(df, BIN + ["index_fairness", "index_willing_change"])
    print("Table A15, paper's specification (country FE, weights, HC1)")
    show(a15_tab)
    add = pd.DataFrame({"impacts": a15_tab["Climate impacts"],
                        "policies": a15_tab["Climate policies"],
                        "sum": a15_tab["Climate impacts"] + a15_tab["Climate policies"],
                        "both_actual": a15_tab["Both"]})
    add["gap"] = add["both_actual"] - add["sum"]
    print("Additivity of the combined arm")
    show(add.round(4))

    # ---- 5. balance ---------------------------------------------------------------------
    print("\n5. Balance: covariate means by arm, joint F-test")
    show(by_arm(df, COVARIATES))
    bal = balance(df, COVARIATES)
    show(bal)

    # ---- 6. data quality ----------------------------------------------------------------
    print(f"\n6. Data quality. Rows {len(df):,} | Countries {df[cfg.COUNTRY].nunique()} | "
          f"exact duplicate rows {df.duplicated().sum():,}")
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    print("Missingness (%), variables with any:")
    print(miss[miss > 0].round(2).to_string())
    print("Constant columns (post-screening sample):",
          [c for c in df.columns if df[c].nunique(dropna=True) == 1] or "none")
    print("Share missing rush_treatment:", round(df.rush_treatment.isna().mean(), 4),
          "| control share:", round((df[cfg.ARM] == 1).mean(), 4), "(structural)")
    mb = df.groupby(cfg.ARM)[df.columns.drop(cfg.ARM)].apply(lambda g: g.isna().mean())
    spread = ((mb.max() - mb.min()) * 100).sort_values(ascending=False)
    print("Differential item non-response across arms (max - min, pp) above 0.5:")
    print(spread[spread > 0.5].round(2).to_string() or "  none")
    for var, arms in (("watched_climate", (2, 4)), ("watched_policy", (3, 4))):
        s = df.loc[df[cfg.ARM].isin(arms), var].replace("", np.nan)
        print(f"--- {var}, arms {arms} (N={len(s):,}) ---")
        print((s.value_counts(dropna=False) / len(s)).round(4).to_string())
    print("Duration (minutes):")
    print(df["duration"].describe(percentiles=[.5, .9, .99]).round(2).to_string())
    print(f"Rows above 24 hours: {(df['duration'] > 1440).sum():,}")
    w = df[cfg.WEIGHT]
    print(f"Weights: min {w.min():.3f}  max {w.max():.3f}  mean {w.mean():.3f}")

    # ---- 7. export ----------------------------------------------------------------------
    desc.to_latex(table_path("table_descriptives.tex"), float_format="%.3f", na_rep="",
                  caption="Descriptive statistics", label="tab:desc")
    a15_tab.to_latex(table_path("table_a15_country_fe.tex"), float_format="%.4f", na_rep="",
                     caption="Treatment effects on support for climate policies", label="tab:a15_country_fe")
    desc.to_csv(table_path("table_descriptives.csv"))
    print("wrote table_descriptives.tex/.csv, table_a15_country_fe.tex")

    return {"desc": desc, "summary": summary, "table_a15": a15_tab, "balance": bal}

if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("summary_stats")
