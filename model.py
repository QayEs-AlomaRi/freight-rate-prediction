# comparing models. train on Jan-Aug and test on Sep-Oct
#
# results (MAE):
#   baseline                      $104.73
#   random forest, dollars         $96.39
#   random forest, rate per mile   $89.73  <- best, used in predict.py
#   with a day counter            ~$114.60 <- worse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from prep import load_and_clean, make_features, align

SPLIT = "2025-09-01"

# use the last 2 months as test data since we need to predict future months
df = load_and_clean("data/train_test.csv")
train = df[df.date < SPLIT].copy()
valid = df[df.date >= SPLIT].copy()
y = valid.posted_rate

print(f"train rows: {len(train):,} ({train.date.min().date()} to {train.date.max().date()})")
print(f"test rows: {len(valid):,} ({valid.date.min().date()} to {valid.date.max().date()})")


def forest():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1)


# simple baseline to compare with: median rate per mile for each equipment and distance range, then multiply by distance
bands = [0, 500, 1000, 1500, 2500, 99999]
train["band"] = pd.cut(train.distance, bands)
valid["band"] = pd.cut(valid.distance, bands)
lookup = train.groupby(["equipment", "band"], observed=True).rate_per_mile.median()
fallback = train.rate_per_mile.median()

baseline = valid.apply(
    lambda r: lookup.get((r.equipment, r.band), fallback) * r.distance, axis=1)
print(f"\nbaseline MAE: ${mean_absolute_error(y, baseline):,.2f}")

X_train = make_features(train)
X_valid = align(make_features(valid), X_train)

# model 1: predict the price directly
dollars = forest().fit(X_train, train.posted_rate)
print(f"random forest (price) MAE: ${mean_absolute_error(y, dollars.predict(X_valid)):,.2f}")

print("\nfeature importance (%):")
print((pd.Series(dollars.feature_importances_, index=X_train.columns)
         .sort_values(ascending=False) * 100).round(1).head(6).to_string())

# model 2: distance is almost all of the importance in model 1, so I tried predicting rate per mile and then multiplying by distance. this was better
per_mile = forest().fit(X_train, train.rate_per_mile)
preds = per_mile.predict(X_valid) * valid.distance
print(f"\nrandom forest (rate per mile) MAE: ${mean_absolute_error(y, preds):,.2f}")

