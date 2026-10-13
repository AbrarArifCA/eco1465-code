"""
Classify the open-ended `cc_field` responses with GPT-6 Luna.

Two dimensions are extracted, not one (see NOTE ON DESIGN below):
    concern  -- does the respondent treat climate change as a real problem?
    action   -- does the respondent want government to act?
plus `substantive`, which separates blank / "don't know" from genuine ambivalence.

Pipeline:
    python -m Scripts.cc_classify extract            # all.dta -> cc_data.parquet
    python -m Scripts.cc_classify validate --n 600   # score against the 9,374 human labels
    python -m Scripts.cc_classify validate --n 600 --reasoning medium   # compare effort levels
    python -m Scripts.cc_classify run                # classify everything (resumable)
    python -m Scripts.cc_classify assemble           # -> cc_labels.csv, ready to merge

NOTE ON DESIGN
The survey asked two things at once: "what are your main considerations?" and
"what should [Country] government do?". A respondent can be worried but opposed
to government action (and vice versa). The authors' `cc_field_worry` flag folds
these together. Keeping them apart costs nothing and is more defensible; collapse
them afterwards if you want a single label comparable to theirs.

COST. At gpt-6-luna's $0.10/1M input and $0.50/1M output, classifying all ~35,700
non-blank responses one-per-call runs roughly $3. Cost is not the binding
constraint, so this does NOT batch several responses into one prompt -- batching
saves a couple of dollars and introduces response-misalignment bugs.

REASONING EFFORT. gpt-6-luna is a reasoning model and bills reasoning tokens as
output. This task is short-text classification against a fixed codebook, which
needs no extended deliberation, so the default here is "low". Raising it inflates
cost and latency for little gain -- but try "medium" on the validation sample and
compare the confusion matrices before assuming that. Set "none" for the cheapest
possible run.
"""

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "gpt-6-luna"
PRICE_IN, PRICE_OUT = 0.10, 0.50        # USD per 1M tokens
REASONING = "low"                        # none | low | medium | high | xhigh | max
# Paths: inside the repository, read all.dta from config.py and write to output/cc_work/;
# run on its own (python cc_classify.py ...), use all.dta and cc_work/ in the current folder.
try:
    import config as _cfg
    from Scripts.data import data_path as _data_path
    DTA = str(_data_path())
    WORK = _cfg.OUTPUT_DIR / "cc_work"
except Exception:
    DTA = "all.dta"
    WORK = Path("cc_work")
WORK.mkdir(parents=True, exist_ok=True)
CACHE = WORK / "cache.jsonl"          # resumable: one JSON object per respondent

HUMAN_FLAGS = ["cc_field_worry", "cc_field_no_worry", "cc_field_empty",
               "cc_field_do_not_know", "cc_field_ambiguous"]

# --------------------------------------------------------------------------
# 1. EXTRACT
# --------------------------------------------------------------------------
def extract():
    cols = (["cc_field", "cc_field_english", "length_cc_field", "country",
             "treatment", "weight"] + HUMAN_FLAGS)
    df = pd.read_stata(DTA, columns=cols, convert_categoricals=False)
    df = df.reset_index().rename(columns={"index": "rid"})
    df["cc_field"] = df["cc_field"].fillna("").astype(str).str.strip()
    df["labelled"] = df["cc_field_worry"].notna()
    df.to_parquet(WORK / "cc_data.parquet")
    print(f"{len(df):,} rows | non-blank cc_field: {(df.cc_field != '').sum():,} "
          f"| human-labelled: {df.labelled.sum():,}")
    return df


# --------------------------------------------------------------------------
# 2. PROMPT + SCHEMA
# --------------------------------------------------------------------------
SYSTEM = """You code open-ended survey responses about climate change for an academic study.

Respondents in 20 countries answered, in their own language:
"When thinking about climate change, what are your main considerations?
 What should the [COUNTRY] government do regarding climate change?"

Read the response IN ITS ORIGINAL LANGUAGE. Do not translate it first; judge it directly.

Code four fields.

substantive: true if the response expresses any view about climate change or policy.
  false for blanks, gibberish, keyboard mashing, "idk", "no comment", "nothing",
  or text that answers some other question entirely. A short response can still be
  substantive ("planting trees" is substantive; "asdfgh" is not).

concern: the respondent's view of climate change as a problem.
  "concerned"   - treats it as real and serious, or describes harms
  "dismissive"  - denies it, calls it exaggerated, a hoax, natural, or not a priority
  "neutral"     - mentions it without evaluating, or is genuinely balanced
  "unclear"     - not substantive, or impossible to tell

action: the respondent's view on government action.
  "pro"      - wants government to act, or names policies it should adopt
  "anti"     - opposes action, warns about cost, or says other issues come first
  "neutral"  - no view expressed on what government should do
  "unclear"  - not substantive

intensity: -2 to +2 for the OVERALL stance combining both
  (+2 strongly supportive of climate action, 0 genuinely neutral,
   -2 strongly opposed/dismissive). Use 0 when not substantive.

evidence: up to 8 words quoted from the original response that drove your coding,
  or "" if not substantive.

Judge only what is written. Do not infer from the country. A respondent who is
worried but opposes government spending is concerned=concerned, action=anti --
that combination is real and you should code it when you see it."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["substantive", "concern", "action", "intensity", "evidence"],
    "properties": {
        "substantive": {"type": "boolean"},
        "concern": {"type": "string",
                    "enum": ["concerned", "dismissive", "neutral", "unclear"]},
        "action": {"type": "string",
                   "enum": ["pro", "anti", "neutral", "unclear"]},
        "intensity": {"type": "integer", "minimum": -2, "maximum": 2},
        "evidence": {"type": "string"},
    },
}


def user_msg(text, country):
    return f"Country: {country}\n\nResponse:\n\"\"\"\n{text}\n\"\"\""


# --------------------------------------------------------------------------
# 3. API CALL
# --------------------------------------------------------------------------
def make_client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    return OpenAI()


# Set False if the API rejects reasoning_effort on this endpoint; the first such
# error flips it automatically and the call is retried without the parameter.
_USE_REASONING = True


def classify_one(client, rid, text, country, retries=5):
    """One respondent per call.

    NOTE: temperature/seed are deliberately NOT passed -- the GPT-6 docs do not
    list them for this model, and sending an unsupported parameter is an error.
    That removes the usual lever for reproducibility, so record the model ID and
    the date of your run, and keep cache.jsonl: it IS your reproducibility
    artefact. Re-running later may not reproduce identical codings."""
    global _USE_REASONING
    for attempt in range(retries):
        try:
            kw = {"reasoning_effort": REASONING} if _USE_REASONING else {}
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user_msg(text, country)}],
                response_format={"type": "json_schema", "json_schema":
                                 {"name": "cc_coding", "strict": True,
                                  "schema": SCHEMA}},
                **kw,
            )
            out = json.loads(r.choices[0].message.content)
            out["rid"] = int(rid)
            out["_in"] = r.usage.prompt_tokens
            out["_out"] = r.usage.completion_tokens
            return out
        except Exception as e:
            msg = str(e)
            if _USE_REASONING and "reasoning_effort" in msg:
                print("NOTE: reasoning_effort rejected; continuing without it.")
                _USE_REASONING = False
                continue
            if attempt == retries - 1:
                return {"rid": int(rid), "error": msg[:200]}
            time.sleep(2 ** attempt)


def run_batch(df, workers=8, limit=None):
    """Classify df rows, skipping anything already in the cache. Safe to
    interrupt and re-run -- results append to cache.jsonl as they land."""
    done = set()
    if CACHE.exists():
        with open(CACHE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["rid"])
                except Exception:
                    pass
    todo = df[~df.rid.isin(done) & (df.cc_field != "")]
    if limit:
        todo = todo.head(limit)
    print(f"cached {len(done):,} | to do {len(todo):,}")
    if todo.empty:
        return

    client = make_client()
    tin = tout = 0
    with open(CACHE, "a") as fh, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(classify_one, client, r.rid, r.cc_field, r.country): r.rid
                for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            tin += res.pop("_in", 0); tout += res.pop("_out", 0)
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                fh.flush()
                cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
                print(f"  {i:,}/{len(todo):,}  ~${cost:,.2f}")
    print(f"done. ~${tin/1e6*PRICE_IN + tout/1e6*PRICE_OUT:,.2f}  ({tout:,} output tokens incl. reasoning)")


# --------------------------------------------------------------------------
# 4. VALIDATE against the 9,374 human labels
# --------------------------------------------------------------------------
def human_label(row):
    """Collapse the authors' five flags into one gold label.
    CAUTION: the flags are NOT a partition -- 13.7% of labelled rows have all
    five set to zero and 84 have more than one. Those become 'other', excluded
    from the headline accuracy and reported separately."""
    if row.cc_field_empty == 1:
        return "not_substantive"
    if row.cc_field_do_not_know == 1:
        return "not_substantive"
    if row.cc_field_worry == 1 and row.cc_field_no_worry != 1:
        return "supportive"
    if row.cc_field_no_worry == 1 and row.cc_field_worry != 1:
        return "skeptical"
    if row.cc_field_ambiguous == 1:
        return "indifferent"
    return "other"


def pred_label(rec):
    """Collapse the model output into the same 3+1 scheme."""
    if not rec.get("substantive", False):
        return "not_substantive"
    if rec["intensity"] >= 1:
        return "supportive"
    if rec["intensity"] <= -1:
        return "skeptical"
    return "indifferent"


def validate(n=600, workers=8):
    df = pd.read_parquet(WORK / "cc_data.parquet")
    lab = df[df.labelled].copy()
    lab["gold"] = lab.apply(human_label, axis=1)

    # Stratify, and OVERSAMPLE skeptics: they are 5.3% of the labelled data, so a
    # random 600 yields ~32 of them -- far too few to estimate recall on the class
    # you most care about.
    quota = {"supportive": int(n * .40), "skeptical": int(n * .30),
             "indifferent": int(n * .15), "not_substantive": int(n * .15)}
    parts = [g.sample(min(len(g), quota.get(k, 0)), random_state=42)
             for k, g in lab.groupby("gold") if k in quota]
    samp = pd.concat(parts)
    print("validation sample:\n", samp.gold.value_counts().to_string(), "\n")

    run_batch(samp, workers=workers)

    recs = {}
    with open(CACHE) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                recs[r["rid"]] = r
    samp = samp[samp.rid.isin(recs)].copy()
    if samp.empty:
        sys.exit("No cached results overlap the validation sample. If the run was "
                 "interrupted or every call errored, inspect cc_work/cache.jsonl "
                 "for 'error' entries and re-run `validate`.")
    n_err = sum(1 for l in open(CACHE) if '"error"' in l)
    if n_err:
        print(f"WARNING: {n_err} cached rows are API errors and are excluded.\n")
    samp["pred"] = [pred_label(recs[r]) for r in samp.rid]

    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(samp.gold, samp.pred, zero_division=0))
    order = ["supportive", "indifferent", "skeptical", "not_substantive"]
    cm = confusion_matrix(samp.gold, samp.pred, labels=order)
    print("\nconfusion (rows = human, cols = model)")
    print(pd.DataFrame(cm, index=order, columns=order).to_string())

    samp["evidence"] = [recs[r].get("evidence", "") for r in samp.rid]
    bad = samp[samp.gold != samp.pred]
    bad[["rid", "country", "gold", "pred", "evidence", "cc_field"]].to_csv(
        WORK / "disagreements.csv", index=False)
    print(f"\n{len(bad)} disagreements -> {WORK/'disagreements.csv'}  <- READ THESE")


# --------------------------------------------------------------------------
# 5. ASSEMBLE
# --------------------------------------------------------------------------
def assemble():
    df = pd.read_parquet(WORK / "cc_data.parquet")
    recs = [json.loads(l) for l in open(CACHE)]
    r = pd.DataFrame([x for x in recs if "error" not in x]).drop_duplicates("rid")

    out = df[["rid", "country", "treatment", "weight"]].merge(r, on="rid", how="left")
    out["cc_substantive"] = out["substantive"].fillna(False).astype(int)
    out["cc_stance"] = out["intensity"] / 2.0                     # continuous, [-1,1]
    out.loc[out.cc_substantive == 0, "cc_stance"] = np.nan
    out["cc_concern"] = out["concern"]
    out["cc_action"] = out["action"]
    out["cc_3way"] = np.select(
        [out.cc_substantive == 0, out.cc_stance >= 0.5, out.cc_stance <= -0.5],
        ["not_substantive", "supportive", "skeptical"], default="indifferent")

    keep = ["rid", "cc_substantive", "cc_stance", "cc_concern", "cc_action",
            "cc_3way", "evidence"]
    out[keep].to_csv(WORK / "cc_labels.csv", index=False)
    print(out.cc_3way.value_counts(dropna=False).to_string())
    print(f"\nwrote {WORK / 'cc_labels.csv'} ({len(out):,} rows) -- merge on rid "
          f"(= row order of all.dta)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["extract", "validate", "run", "assemble"])
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reasoning", default=REASONING,
                   choices=["none","low","medium","high","xhigh","max"])
    a = p.parse_args()
    REASONING = a.reasoning
    print(f"model={MODEL}  reasoning_effort={REASONING}")

    if a.cmd == "extract":
        extract()
    elif a.cmd == "validate":
        validate(a.n, a.workers)
    elif a.cmd == "run":
        run_batch(pd.read_parquet(WORK / "cc_data.parquet"), a.workers, a.limit)
    elif a.cmd == "assemble":
        assemble()
