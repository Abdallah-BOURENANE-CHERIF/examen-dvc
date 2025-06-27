import sys
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour / dayofweek / since_start / sin‑cos hour from `timestamp`."""
    if "timestamp" not in df.columns:
        return df

    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.copy()
    df["hour"]        = ts.dt.hour
    df["dayofweek"]   = ts.dt.dayofweek
    df["since_start"] = (ts - ts.min()).dt.total_seconds() / 86_400 


    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df.drop(columns="timestamp")



def main(in_dir: pathlib.Path, params_pkl: pathlib.Path, out_pkl: pathlib.Path) -> None:
    X_train = pd.read_csv(in_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(in_dir / "y_train.csv").squeeze()


    X_train = add_time_features(X_train)
    X_train = X_train.select_dtypes(include=[np.number])

    best_params = joblib.load(params_pkl)
    model = GradientBoostingRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, out_pkl)
    print(f" Model trained and saved → {out_pkl}")


if __name__ == "__main__":
    in_dir, params_pkl, out_pkl = map(pathlib.Path, sys.argv[1:4])
    main(in_dir, params_pkl, out_pkl)