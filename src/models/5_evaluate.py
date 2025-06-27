import sys, pathlib, json, joblib, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

model_pkl, in_dir, metrics_dir = map(pathlib.Path, sys.argv[1:4])
metrics_dir.mkdir(parents=True, exist_ok=True)

model = joblib.load(model_pkl)

X_test = pd.read_csv(in_dir / "X_test_scaled.csv")
y_test = pd.read_csv(in_dir / "y_test.csv").squeeze()

y_pred = model.predict(X_test)
pd.DataFrame(
    {"y_true": y_test, "y_pred": y_pred}
).to_csv(in_dir / "y_pred.csv", index=False)

metrics = {
    "rmse": mean_squared_error(y_test, y_pred, squared=False),
    "r2": r2_score(y_test, y_pred),
}
with open(metrics_dir / "scores.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)