import sys, pathlib, json, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.copy()
    df["hour"]        = ts.dt.hour
    df["dayofweek"]   = ts.dt.dayofweek
    df["since_start"] = (ts - ts.min()).dt.total_seconds() / 86_400
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    return df.drop(columns="timestamp")


model_pkl, data_dir, metrics_dir = map(pathlib.Path, sys.argv[1:4])
metrics_dir.mkdir(parents=True, exist_ok=True)

model  = joblib.load(model_pkl)

X_test = pd.read_csv(data_dir / "X_test_scaled.csv")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

X_test = add_time_features(X_test).select_dtypes(include=[np.number])

y_pred = model.predict(X_test)
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
    data_dir / "y_pred.csv", index=False
)

metrics = {
    "rmse": mean_squared_error(y_test, y_pred, squared=False),
    "r2":   r2_score(y_test, y_pred),
}
(json_path := metrics_dir / "scores.json").write_text(json.dumps(metrics, indent=2))
print(" Metrics written →", json_path)