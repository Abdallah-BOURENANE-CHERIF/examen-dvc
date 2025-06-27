import sys, pathlib, joblib, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string `timestamp` into hour, dayofweek, and cyclic hour."""
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



in_dir, out_pkl = map(pathlib.Path, sys.argv[1:3])

X_train = pd.read_csv(in_dir / "X_train_scaled.csv")
y_train = pd.read_csv(in_dir / "y_train.csv").squeeze()


X_train = add_time_features(X_train)          


X_train = X_train.select_dtypes(include="number")


param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.6, 0.8, 1.0],
}

gbr  = GradientBoostingRegressor(random_state=42)
grid = GridSearchCV(
    gbr,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    error_score="raise",
)
grid.fit(X_train, y_train)

joblib.dump(grid.best_params_, out_pkl)
print(" Best params saved →", out_pkl)