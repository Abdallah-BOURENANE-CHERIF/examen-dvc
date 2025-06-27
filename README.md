First Name: BOURENANE CHERIF
Last Name: Abdallah

e-mail adresse: abdallah.bourenanecherif@amundi.com

Please find the codes in Master Branch of the repository.

Mineral‑Flotation Model Pipeline:


A fully‑reproducible DVC 3.x project that models silica concentration in an
iron‑ore flotation process.  The repo shows how to go from raw CSV → trained
model → evaluation metrics while storing all large artifacts in DagsHub.


.
├── data
│   ├── raw/                
│   └── processed_data/     
├── models/                 
├── metrics/                
├── src/                   
│   ├── data/
│   └── models/
├── dvc.yaml  dvc.lock      
└── README.md

Quick‑start

git clone https://dagshub.com/Abdallah-BOURENANE-CHERIF/examen-dvc
cd examen-dvc

# 2 create virtual‑env & install deps
python -m venv .venv
source .venv/bin/activate     
pip install -r requirements.txt

# 3 download raw data
python src/data/import_raw_data.py -o data/raw   

# 4 track dataset with DVC
dvc init -q                        
dvc add data/raw/raw.csv

# 5 build the pipeline & run everything once
# (dvc.yaml already contains the stages)
dvc repro --force                  

# 6 inspect metrics
cat metrics/scores.json 

