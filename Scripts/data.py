"""Loading all.dta and building the samples and design matrices shared by the sections.
"""
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn.model_selection as skm

import config as cfg

_COLS = {}          # (column, convert_categoricals) -> Series


def data_path():
    for p in cfg.DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "all.dta not found. Put it in data/ (see data/README.md) or edit DATA_CANDIDATES "
        "in config.py. Looked in:\n  " + "\n  ".join(str(p) for p in cfg.DATA_CANDIDATES))


def read(columns, categoricals=False):
    """Columns of all.dta. Numeric codes by default; labels with categoricals=True."""
    missing = [c for c in columns if (c, categoricals) not in _COLS]
    if missing:
        df = pd.read_stata(data_path(), columns=missing, convert_categoricals=categoricals)
        for c in missing:
            _COLS[(c, categoricals)] = df[c]
    return pd.DataFrame({c: _COLS[(c, categoricals)] for c in columns})


def variable_labels():
    with pd.io.stata.StataReader(data_path()) as rdr:
        return rdr.variable_labels()


def dummies(s, base, prefix):
    """One dummy per category, dropping the base category."""
    D = pd.get_dummies(s.astype(str), prefix=prefix, prefix_sep=": ", dtype=float)
    return D.drop(columns=f"{prefix}: {base}")


# -----------------------------------------------------------------------------------------
# Main specification (Reports 3-5): outcome, 3 arm dummies, pre-treatment controls,
# 19 country dummies; unweighted; the 30 rows with missing `urban` dropped (N = 40,650)
# -----------------------------------------------------------------------------------------
@cache
def main_design():
    raw = read([cfg.ARM, cfg.COUNTRY, cfg.OUTCOME] + cfg.PRE, categoricals=True)
    d = raw.dropna(subset=["urban"]).copy()

    y = (d[cfg.OUTCOME] > 0).astype(float).rename("tax_transfers_support_bin")

    X = pd.DataFrame(index=d.index)
    X["Impacts"]  = (d[cfg.ARM] == "Climate impacts").astype(float)
    X["Policies"] = (d[cfg.ARM] == "Climate policy").astype(float)
    X["Both"]     = (d[cfg.ARM] == "Both").astype(float)
    X["female"]   = d["female"].astype(float)
    X["college"]  = (d["college"] == "College Degree").astype(float)   # raw coding is 1/2
    for c in ["urban", "couple", "hh_size", "children", "polluting_sector", "home_owner",
              "wealth", "interested_politics", "car_dependency", "gas_expenses",
              "flights_agg", "frequency_beef", "availability_transport", "hit_by_covid"]:
        X[c] = d[c].astype(float)
    X = pd.concat([X,
                   dummies(d["age"].replace({"Below 18": "18-24"}), "18-24", "age"),
                   dummies(d["income_factor"],  "Q1",      "income"),
                   dummies(d["employment_agg"], "Working", "employment"),
                   dummies(d["econ_leaning"],   "Center",  "econ"),
                   dummies(d[cfg.COUNTRY],        "US",      "country")], axis=1)

    Xa, Y = np.asarray(X), np.asarray(y)
    X_scale = Xa.std(0)
    Xs = (Xa - Xa.mean(0)[None, :]) / X_scale[None, :]
    ctrl = [c for c in X.columns if c not in cfg.TREAT]
    return SimpleNamespace(d=d, X=X, y=y, Xa=Xa, Y=Y, Xs=Xs, X_scale=X_scale,
                           CTRL=ctrl, feature_names=list(X.columns))


def kfold():
    """The K-fold split used by every CV in the full sample (same seed everywhere)."""
    return skm.KFold(cfg.K, shuffle=True, random_state=cfg.SEED)


@cache
def split():
    """75 / 25 train / test split of the main sample, plus the inner CV for tuning."""
    Y = main_design().Y
    tr, te = next(skm.ShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE,
                                   random_state=cfg.SPLIT_SEED).split(Y))
    inner = skm.KFold(n_splits=cfg.K, shuffle=True, random_state=cfg.INNER_SEED)
    return SimpleNamespace(tr=tr, te=te, inner=inner)


# -----------------------------------------------------------------------------------------
# The paper's Table A15 column (3) specification: set A controls, country FE, weights,
# all 40,680 respondents, Left as the omitted leaning
# -----------------------------------------------------------------------------------------
@cache
def a15_design():
    a = read([cfg.ARM, cfg.COUNTRY, cfg.WEIGHT, cfg.OUTCOME] + cfg.SET_A, categoricals=True)
    y = (a[cfg.OUTCOME] > 0).astype(float).rename("tax_transfers_support_bin")
    X = pd.concat([
        pd.DataFrame({"Impacts":  (a[cfg.ARM] == "Climate impacts").astype(float),
                      "Policies": (a[cfg.ARM] == "Climate policy").astype(float),
                      "Both":     (a[cfg.ARM] == "Both").astype(float)}),
        a[["female", "other", "children"]].astype(float),
        dummies(a["age_control"],   "18-24",             "age"),
        dummies(a["income_factor"], "Q1",                "income"),
        dummies(a["educ_categ"],    "Below high-school", "educ"),
        dummies(a["econ_leaning"],  "Left",              "econ"),      # paper's omitted category
        dummies(a[cfg.COUNTRY],       "US",                "country"),
    ], axis=1)
    return SimpleNamespace(X=X, y=y, w=a[cfg.WEIGHT].to_numpy())


# -----------------------------------------------------------------------------------------
# Project 1 frame: all respondents, labelled categories, derived indicators
# -----------------------------------------------------------------------------------------
PROJECT_COLS = ["treatment", "country", "weight", "tax_transfers_support", "standard_support",
                "investments_support", "index_fairness", "index_willing_change",
                "index_main_policies", "female", "other", "children", "age_control",
                "income_factor", "educ_categ", "econ_leaning", "interested_politics",
                "watched_climate", "watched_policy", "donation_fraction", "petition",
                "index_lose_policies_subjective", "index_policies_effective", "urban",
                "car_dependency", "availability_transport", "college"]


@cache
def project_frame():
    d = read(PROJECT_COLS, categoricals=True).copy()
    d["arm"] = d.treatment.astype(str)
    for k, v in [("Impacts", "Climate impacts"), ("Policies", "Climate policy"), ("Both", "Both")]:
        d[k] = (d.arm == v).astype(float)
    d["ctrl"] = (d.arm == "None").astype(float)
    for p in ["tax_transfers", "standard", "investments"]:
        d[f"{p}_bin"] = (d[f"{p}_support"] > 0).astype(float)
    d["college01"] = (d.college == "College Degree").astype(float)
    d["age50"] = (d.age_control == "50+").astype(float)
    d["left"] = d.econ_leaning.isin(["Left", "Very left"]).astype(float)
    d["right"] = d.econ_leaning.isin(["Right", "Very right"]).astype(float)
    d["incQ4"] = (d.income_factor == "Q4").astype(float)
    d["donate_any"] = (d.donation_fraction > 0).astype(float)
    return d


# -----------------------------------------------------------------------------------------
# Report 5 sample: control vs. climate-policies arm, with S (interest in politics) and the
# post-treatment mediator M (believes own household would lose)
# -----------------------------------------------------------------------------------------
@cache
def dag_sample():
    md = main_design()
    d = md.d
    df = pd.DataFrame({"Y": md.y, "S": d["interested_politics"].astype(float),
                       "arm": d[cfg.ARM].astype(str), "country": d[cfg.COUNTRY]})
    med = read(["index_lose_policies_subjective", "index_policies_effective"])
    df = df.join(med)                                       # same row index as d
    exp = df[df["arm"].isin(["None", "Climate policy"])].copy()
    exp["D"] = (exp["arm"] == "Climate policy").astype(float)
    return exp.rename(columns={"index_lose_policies_subjective": "M"})
