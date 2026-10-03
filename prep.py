import pandas as pd

# rows with a rate per mile higher than this get removed (bad data)
MAX_RATE_PER_MILE = 6.0

# columns used by the model
FEATURES = ["distance", "weight", "market_index", "quote_signal",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

# medians from the training data, so I can use the same ones for validation
_medians = {}


def load_and_clean(path, is_training=True):
    df = pd.read_csv(path, parse_dates=["date"])

    # some weights are negative which doesnt make sense. the numbers look normal without the minus so I just take the absolute value
    df["weight"] = df["weight"].abs()

    # save the medians only from training data
    if is_training:
        _medians["weight"] = df["weight"].median()
        _medians["market_index"] = df["market_index"].median()
    elif not _medians:
        raise RuntimeError("load the training data first to get the medians")

    # fill missing values with the median (cant drop rows in validation)
    for col in ("weight", "market_index"):
        df[col] = df[col].fillna(_medians[col])

    if is_training:
        df["rate_per_mile"] = df["posted_rate"] / df["distance"]
        # most loads are between $1.70 and $3.20 per mile, some are way higher (up to $10+) so I remove those outliers from training
        df = df[df["rate_per_mile"] < MAX_RATE_PER_MILE]

    return df


def make_features(df):
    # using lat/lon instead of city names because some cities in validation are not in the training data
    X = df[FEATURES + ["equipment"]].copy()
    X["dayofweek"] = df["date"].dt.dayofweek
    # turn equipment into 0/1 columns
    return pd.get_dummies(X, columns=["equipment"])


# make sure the columns are the same as training (missing ones become 0)
def align(X, reference):
    return X.reindex(columns=reference.columns, fill_value=0)
