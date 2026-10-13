"""
Configuration file for the notebook code. Contains paths, seeds, and variable names
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------------------
DATA_CANDIDATES = ROOT / "data" / "all.dta",


# ---------------------------------------------------------------------------------------
# Set up output directories for tables, figures, and results
# ---------------------------------------------------------------------------------------
OUTPUT_DIR  = ROOT / "output"
TABLES_DIR  = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"     # cached numbers passed between sections

# ---------------------------------------------------------------------------------------
# Seeds and sample split (to maintain comparability across sections)
# ---------------------------------------------------------------------------------------
SEED        = 0       # Maintain same seed
K           = 5       # Use 5 folds for cross-validation
TEST_SIZE   = 0.25    # train / test split for test-MSE comparisons
SPLIT_SEED  = 1       # ShuffleSplit seed for that split
INNER_SEED  = 2       # KFold seed for CV nested inside the training sample

# ---------------------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------------------
ARM  = "treatment"
COUNTRY = "country"
WEIGHT = "weight"

ARM_LABELS = {1: "Control", 2: "Climate impacts", 3: "Climate policies", 4: "Both"}
OUTCOME = "tax_transfers_support"          # Support for carbon tax with rebate. Binary variable with 0 as no support and 1 for support

# Pre-treatment controls of the main specification
PRE = ["female", "age", "income_factor", "college", "urban", "couple", "hh_size", "children",
       "employment_agg", "polluting_sector", "home_owner", "wealth", "interested_politics",
       "econ_leaning", "car_dependency", "gas_expenses", "flights_agg", "frequency_beef",
       "availability_transport", "hit_by_covid"]

# Variables for Table A15
SET_A = ["female", "other", "children", "age_control", "income_factor", "educ_categ",
         "econ_leaning"]

TREAT = ["Impacts", "Policies", "Both"]
