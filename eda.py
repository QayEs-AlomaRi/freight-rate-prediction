# exploring the data before building the model (only prints, nothing saved)
import pandas as pd

tr = pd.read_csv("data/train_test.csv", parse_dates=["date"])
va = pd.read_csv("data/validation.csv", parse_dates=["date"])
dec = pd.read_csv("data/december_chart_inputs.csv")
target = set(tr.columns) - set(va.columns)
tr["rate_per_mile"] = tr.posted_rate / tr.distance


def section(title):
    print(f"\n{'=' * 55}\n{title}\n{'=' * 55}")


# check the columns in each file
section("COLUMNS")
print("train:", tr.shape, "| validation:", va.shape)
print("column to predict:", target)
print("columns missing from the december file:",
      sorted(set(va.columns) - set(dec.columns) - {"load_id"}))

# check the dates
section("DATES")
for name, df in [("train", tr), ("validation", va)]:
    print(f"{name:11s} {df.date.min().date()} -> {df.date.max().date()}")
print("\ndates dont overlap so I should split by date not randomly")

# check for missing values and negative weights
section("MISSING VALUES")
print(pd.DataFrame({"train": tr.isna().sum(), "valid": va.isna().sum()})
        .query("train > 0 or valid > 0").to_string())
print(f"\nnegative weights: {(tr.weight < 0).sum()} train, {(va.weight < 0).sum()} validation")
print("negative weights without minus: %.0f to %.0f (normal: %.0f to %.0f)" % (
    tr.weight[tr.weight < 0].abs().min(), tr.weight[tr.weight < 0].abs().max(),
    tr.weight[tr.weight > 0].min(), tr.weight[tr.weight > 0].max()))

# look at rate per mile to find outliers
section("RATE PER MILE")
print(tr.rate_per_mile.quantile([0.01, 0.5, 0.99, 0.999]).round(2).to_string())
for cut in (4, 6, 10):
    print(f"above ${cut}/mile: {(tr.rate_per_mile > cut).sum():>4d} rows")
print("\nhighest ones:")
print(tr.nlargest(5, "rate_per_mile")[
    ["distance", "weight", "equipment", "posted_rate", "rate_per_mile"]].round(2).to_string())

# which columns are related to the price
section("CORRELATION")
num = ["distance", "weight", "market_index", "quote_signal"]
print(pd.DataFrame({
    "vs posted_rate": tr[num + ["posted_rate"]].corr()["posted_rate"].drop("posted_rate"),
    "vs rate_per_mile": tr[num + ["rate_per_mile"]].corr()["rate_per_mile"].drop("rate_per_mile"),
}).round(3).to_string())
print("\naverage rate per mile by equipment:")
print(tr.groupby("equipment").rate_per_mile.mean().round(3).to_string())
print("\ncities in validation that are not in train:",
      sorted(set(va.pickup) - set(tr.pickup)))

# check if prices change by month
section("BY MONTH")
print(tr.groupby(tr.date.dt.to_period("M")).agg(
    rate=("posted_rate", "mean"),
    rpm=("rate_per_mile", "mean"),
    market=("market_index", "mean")).round(3).to_string())
print("\nvalidation market_index by month:")
print(va.groupby(va.date.dt.to_period("M")).market_index.mean().round(3).to_string())