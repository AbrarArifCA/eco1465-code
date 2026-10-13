"""Runs the section scripts from main.ipynb and passes results between them.

Runner function to enable and disable specific sections from running in main notebook
 
"""
import importlib
import pickle
import time

import config as cfg

# name -> module, in the order of the paper / notebook
SECTIONS = {
    "summary_stats":       "s01_summary_stats",
    "descriptives":        "s02_descriptives",
    "balance":             "s03_balance",
    "compliance":          "s04_compliance",
    "table_a15":           "s05_table_a15",
    "ols":                 "s06_ols",
    "ridge":               "s07_ridge",
    "lasso":               "s08_lasso",
    "compare_linear":      "s09_compare_linear",
    "double_lasso":        "s10_double_lasso",
    "a15_penalized":       "s11_a15_penalized",
    "regression_tree":     "s12_regression_tree",
    "bagging_rf":          "s13_bagging_rf",
    "boosting":            "s14_boosting",
    "error_comparison":    "s15_error_comparison",
    "dags":                "s16_dags"
}


class MissingResults(RuntimeError):
    pass


class Context:
    """What a section receives: settings, and the results of the other sections."""

    def __init__(self):
        self.cfg = cfg
        self.results = {}

    def need(self, section, key=None):
        if section not in self.results:
            f = cfg.RESULTS_DIR / f"{section}.pkl"
            if not f.exists():
                raise MissingResults(
                    f"This section uses results from '{section}', which has not been run yet. "
                    f"Set RUN['{section}'] = True once (its results are then saved to "
                    f"{f.name} and it can be switched off again).")
            with open(f, "rb") as fh:
                self.results[section] = pickle.load(fh)
            print(f"   (using saved results of '{section}')")
        r = self.results[section]
        return r if key is None else r[key]


class Runner:
    def __init__(self, run_flags):
        unknown = set(run_flags) - set(SECTIONS)
        if unknown:
            raise KeyError(f"Unknown section(s) in RUN: {sorted(unknown)}")
        self.flags = run_flags
        self.ctx = Context()
        self.timings = {}

    def __call__(self, name):
        if not self.flags.get(name, False):
            print(f"[skipped] {name}  (RUN['{name}'] is False)")
            return None
        mod = importlib.import_module(f"Scripts.{SECTIONS[name]}")
        print(f"[running] {name}: {mod.TITLE}")
        t0 = time.time()
        out = mod.run(self.ctx) or {}
        self.ctx.results[name] = out
        with open(cfg.RESULTS_DIR / f"{name}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        self.timings[name] = time.time() - t0
        print(f"[done]    {name} in {self.timings[name]:.0f}s")
        return out


def run_standalone(name):
    """`python -m Scripts.<module>`: run one section from the terminal."""
    import matplotlib
    matplotlib.use("Agg")
    Runner({name: True})(name)
