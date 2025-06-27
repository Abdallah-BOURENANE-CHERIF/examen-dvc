

import sys, pathlib, pandas as pd, requests, io

URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

out_file = pathlib.Path(sys.argv[1]) 

out_file.parent.mkdir(parents=True, exist_ok=True)

csv_bytes = requests.get(URL, timeout=30).content
df = pd.read_csv(io.BytesIO(csv_bytes))
df.to_csv(out_file, index=False)

print(f"Saved → {out_file}")

