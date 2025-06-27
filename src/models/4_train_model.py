import sys, pathlib, joblib, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

in_dir, params_pkl, out_pkl = map(pathlib.Path, sys.argv[1:4])

X_train = pd.read_csv(in_dir / "X_train_scaled.csv")
y_train = pd.read_csv(in_dir / "y_train.csv").squeeze()
best_params = joblib.load(params_pkl)

model = GradientBoostingRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, out_pkl)