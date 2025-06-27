import sys, pathlib, joblib, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

in_dir, out_file = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
X_train = pd.read_csv(in_dir / "X_train_scaled.csv")
y_train = pd.read_csv(in_dir / "y_train.csv").squeeze()

param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.6, 0.8, 1.0],
}

gbr = GradientBoostingRegressor(random_state=42)

grid = GridSearchCV(
    gbr,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

joblib.dump(grid.best_params_, out_file)
print("Best params:", grid.best_params_)