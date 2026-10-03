# train the final model and make the prediction files
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from prep import load_and_clean, make_features, align

# train on all the data this time (Jan-Oct), Sep-Oct are the closest months to Nov-Dec so I want the model to learn from them too
train = load_and_clean("data/train_test.csv")
X_train = make_features(train)
model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
model.fit(X_train, train.rate_per_mile)
print(f"model trained on {len(train):,} rows")

# predict validation data (is_training=False so no rows get removed)
val = load_and_clean("data/validation.csv", is_training=False)
val["predicted_rate"] = model.predict(align(make_features(val), X_train)) * val.distance

# put the predictions in the template and save
template = pd.read_csv("data/validation_predictions_template.csv")
out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id")
assert len(out) == 12_000 and (out.predicted_rate > 0).all()
out.to_csv("validation_predictions.csv", index=False)
print(f"saved {len(out):,} predictions to validation_predictions.csv")

# december chart
# the december file is missing some columns the model needs, so:
# - lat/lon: get them from the training data using the city name
# - market_index and quote_signal: average of each day from the december loads in validation.csv
raw = pd.read_csv("data/train_test.csv")
coords = pd.concat([
    raw[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    raw[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
]).drop_duplicates("city").set_index("city")

december = val[val.date.dt.month == 12].groupby(val.date.dt.normalize())
market = december[["market_index", "quote_signal"]].mean()

chart = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])
for side in ("pickup", "delivery"):
    chart[f"{side}_lat"] = chart[side].map(coords.lat)
    chart[f"{side}_lon"] = chart[side].map(coords.lon)
for col in ("market_index", "quote_signal"):
    chart[col] = chart.date.map(market[col])
assert chart[["pickup_lat", "market_index"]].notna().all().all()

chart["predicted_rate"] = (model.predict(align(make_features(chart), X_train))
                           * chart.distance).round(2)

# save only the original 7 columns, score.py doesnt accept extra ones
chart["date"] = chart.date.dt.strftime("%Y-%m-%d")
chart[["pickup", "delivery", "distance", "equipment", "weight", "date",
       "predicted_rate"]].to_csv("data/december_chart_inputs.csv", index=False)
print(f"december predictions: ${chart.predicted_rate.min():,.2f} to ${chart.predicted_rate.max():,.2f}")