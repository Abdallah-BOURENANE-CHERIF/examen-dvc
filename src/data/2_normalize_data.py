import sys, pathlib, joblib, pandas as pd
from sklearn.preprocessing import StandardScaler

in_dir, out_dir = map(pathlib.Path, sys.argv[1:3])
out_dir.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(in_dir / "X_train.csv")
X_test  = pd.read_csv(in_dir / "X_test.csv")


num_cols = X_train.select_dtypes(include="number").columns
non_num_cols = X_train.columns.difference(num_cols)

scaler = StandardScaler().fit(X_train[num_cols])
joblib.dump({"scaler": scaler, "num_cols": list(num_cols)}, out_dir / "scaler.pkl")


X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[num_cols] = scaler.transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

X_train_scaled.to_csv(out_dir / "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(out_dir / "X_test_scaled.csv", index=False)