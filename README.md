# Freight Rate Prediction

Random forest predicting rate per mile, multiplied by distance. Validated on a
date-based holdout: **$89.73 MAE**, against $104.73 for a lookup-table baseline.

## Setup

Needs Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux
```

```bash
pip install -r requirements.txt
```

The four provided CSVs are already in `data/`.

## Run

```bash
python eda.py        # exploration
python model.py      # model selection on the holdout
python predict.py    # writes both output files
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`predict.py` fills the `predicted_rate` column of `data/december_chart_inputs.csv`.
It can be run again, it just overwrites that column. The chart ends up in
`scorer_results/candidate_december.png`.

| | |
|---|---|
| `prep.py` | cleaning, imputation, feature building |
| `eda.py` | exploration |
| `model.py` | baseline vs two target framings |
| `predict.py` | final model, both output files |
| `score.py` | provided, unmodified |

## Approach

**Split by date.** Training data ends 31 October and the loads to predict start
1 November, so this is forecasting. `model.py` trains on Jan-Aug and scores on
Sep-Oct, months it has not seen. A random split would train on October and
score on March, so the MAE would look better than it really is.
`predict.py` then retrains on all ten months, since the holdout was only for
comparing models and Sep-Oct are the closest data to Nov-Dec.

**Cleaning.** Three issues, all in `prep.py`:

- 292 training and 145 validation rows have negative weights. The absolute
  values span the same 5,000-47,500 lb range as valid rows and price normally,
  so it's a sign error and `abs()` recovers them.
- `weight` and `market_index` have gaps in both files, so rows can't be
  dropped. Filled with training medians. Median rather than mean because the
  sign errors drag the mean (31,029 vs 31,417 after the fix) and barely move
  the median (31,436 vs 31,496).
- Rate per mile normally runs $1.70-3.20. The 99th percentile is $3.18, the
  99.9th $10.07, so the top rows are a separate group. Several ~310-mile
  reefer loads price near $800 while two near-identical ones price above
  $4,000. The 270 training rows above $6/mile are dropped. Validation is left
  alone, since every load still needs a price.

**Features.** distance, weight, market_index, quote_signal, both coordinate
pairs, one-hot equipment, day of week. I used coordinates for routes because
eight cities in validation never appear in training.

**Model.**

| | MAE |
|---|---|
| baseline: median $/mile by equipment × distance band | $104.73 |
| random forest, dollar target | $96.39 |
| random forest, rate-per-mile target | **$89.73** |
| either, plus a day counter | ~$114.60 |

On the dollar target, distance takes 96.6% of feature importance, so the trees
mostly relearn that longer trips cost more. Predicting rate per mile and
multiplying by distance afterwards works better.

I also tried a day counter. Trees can't extrapolate, and training stops at
day 242, so every Sep-Oct row ends up in the same late-August leaf. 0.3%
importance and 20% worse, so I dropped it.

**December chart.** The input file has six features where the model needs ten.
Coordinates come from a city lookup built from the training data. `market_index`
and `quote_signal` are averaged per date from the ~200 real December loads
already in `validation.csv` (only feature columns, no prices). The line runs
$769-$793. Route and equipment are fixed, so the ~3% movement comes from the
market numbers and day of week.