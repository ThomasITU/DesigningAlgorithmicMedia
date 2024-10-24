# Designing Algorithmic Media

Investigate whether increase of polarization and a narrowing skew of opionions is correlated

## Dataset

Initial data set found on the [World Values Survey joint 2017](https://www.worldvaluessurvey.org/WVSEVSjoint2017.jsp)

See handbook for feature mappings [variable correspondece.xlsx](data/Handbooks/EVS_WVS_Joint_v5.0_VariableCorrespondence.xlsx)

We're focused on investigating USA, Germany, Norway and Denmark

## Recommended Installation Instructions

Ensure python and pip is installed then create a virtual environment to keep dependencies clean.

```zsh
python3 -m venv DesigningAlgorithmicMedia
source ~/DesigningAlgorithmicMedia/bin/activate
pip install -r requirements.txt
```

From repository root install git pre-commit hook by running script the script below - To avoid cluttering git history with meta data changes in jupyter notebooks

```zsh
sh ./scripts/install-hooks.sh 
```
