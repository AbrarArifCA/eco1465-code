"""Helpers shared by the section scripts: output paths, display, labels, plotting."""
import numpy as np
import matplotlib.pyplot as plt

import config as cfg

for _d in (cfg.TABLES_DIR, cfg.FIGURES_DIR, cfg.RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- display
def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def show(obj, terminal=True):
    """display() inside Jupyter; print() in a terminal (skipped if terminal=False)."""
    if in_notebook():
        from IPython.display import display
        display(obj)
    elif terminal:
        print(obj)


def show_fig(fig, keep=True):
    """Show a figure in the notebook (no-op in a terminal), or just close it."""
    if keep:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------- outputs
def table_path(name):
    return cfg.TABLES_DIR / name


def fig_path(name):
    return cfg.FIGURES_DIR / name


def save_table(body, name, echo=False):
    p = table_path(name)
    p.write_text(body if body.endswith("\n") else body + "\n")
    print(f"wrote {p.relative_to(cfg.ROOT) if p.is_relative_to(cfg.ROOT) else p}")
    if echo:
        print(body)


def save_fig(fig, name):
    p = fig_path(name)
    fig.savefig(p, bbox_inches="tight")
    print(f"wrote {p.relative_to(cfg.ROOT) if p.is_relative_to(cfg.ROOT) else p}")


def stars(p):
    return "^{***}" if p < .01 else "^{**}" if p < .05 else "^{*}" if p < .1 else ""


# ----------------------------------------------------------------------------- labels
COUNTRY_NAMES = {"AU": "Australia", "BR": "Brazil", "CA": "Canada", "CN": "China",
                 "DE": "Germany", "DK": "Denmark", "FR": "France", "IA": "India",
                 "ID": "Indonesia", "IT": "Italy", "JP": "Japan", "MX": "Mexico",
                 "PL": "Poland", "SA": "South Africa", "SK": "South Korea", "SP": "Spain",
                 "TR": "Turkey", "UA": "Ukraine", "UK": "United Kingdom",
                 "US": "United States"}

PRETTY = {"Impacts": "Climate impacts video", "Policies": "Climate policies video",
          "Both": "Both videos", "female": "Female", "college": "College degree",
          "urban": "Urban", "couple": "Lives as a couple", "hh_size": "Household size",
          "children": "Has children", "polluting_sector": "Works in polluting sector",
          "home_owner": "Home owner", "wealth": "Wealth quintile",
          "interested_politics": "Interest in politics", "car_dependency": "Car dependent",
          "gas_expenses": "Gas expenses", "flights_agg": "Flights per year",
          "frequency_beef": "Beef consumption",
          "availability_transport": "Public transport availability",
          "hit_by_covid": "Hit by COVID"}


def pretty(c):
    if c in PRETTY:
        return PRETTY[c]
    for prefix, fmt in [("country: ", lambda v: COUNTRY_NAMES[v]),
                        ("econ: ", lambda v: f"Econ. leaning: {v}"),
                        ("age: ", lambda v: f"Age {v}"),
                        ("income: ", lambda v: f"Income {v}"),
                        ("employment: ", lambda v: f"Employment: {v}")]:
        if c.startswith(prefix):
            return fmt(c[len(prefix):])
    return c


# ----------------------------------------------------------------------------- plots
# Treatment dummies in colour, controls in grey (53 lines are unreadable otherwise)
TCOL = {"Impacts": "#1b9e77", "Policies": "#d95f02", "Both": "#7570b3"}


def path_plot(path, chosen, title, fname):
    """Coefficient paths against -log(lambda), legend placed as in the ISLP lab."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for c in path.columns:
        if c not in TCOL:
            ax.plot(path.index, path[c], color="0.75", lw=0.7)
    for c, col in TCOL.items():
        ax.plot(path.index, path[c], color=col, lw=2,
                label=f"{c} video" if c != "Both" else "Both videos")
    ax.plot([], [], color="0.75", lw=0.7, label="Controls & country FE")
    ax.axvline(-np.log(chosen), c="k", ls="--", lw=1, label=r"CV-chosen $\lambda$")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel(r"$-\log(\lambda)$", fontsize=12)
    ax.set_ylabel("Standardized coefficients", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, fname)
    return fig


def cv_plot(lams, mse_path, chosen, title, fname, null_mse, one_se=None, xlabel=None):
    """CV error with 1-SE bars; dashed = CV minimum, dotted = 1-SE rule, dash-dot = null."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.errorbar(-np.log(lams), mse_path.mean(1), yerr=mse_path.std(1) / np.sqrt(cfg.K),
                c="C0", ecolor="0.6", elinewidth=0.8, capsize=0)
    ax.axvline(-np.log(chosen), c="k", ls="--", lw=1)
    if one_se is not None:
        ax.axvline(-np.log(one_se), c="k", ls=":", lw=1)
    ax.axhline(null_mse, c="0.5", lw=0.7, ls="-.")
    ax.set_xlabel(xlabel or r"$-\log(\lambda)$", fontsize=12)
    ax.set_ylabel("Cross-validated MSE", fontsize=12)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    save_fig(fig, fname)
    return fig
