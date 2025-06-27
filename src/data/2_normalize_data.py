import sys, pathlib, joblib, pandas as pd
from sklearn.preprocessing import StandardScaler

in_dir, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(in_dir / "X_train.csv")
X_test  = pd.read_csv(in_dir / "X_test.csv")

scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, out_dir / "scaler.pkl")

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    out_dir / "X_train_scaled.csv", index=False
)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    out_dir / "X_test_scaled.csv", index=False
)