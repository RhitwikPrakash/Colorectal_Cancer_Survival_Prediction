# DagsHub / MLflow experiment tracking

Set the following environment variables locally before running the MLflow pipeline:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/rhitwik.2005/Colorectal_Cancer_Survival_Prediction.mlflow"
export MLFLOW_TRACKING_USERNAME="rhitwik.2005"
export MLFLOW_TRACKING_PASSWORD="<your-new-DagsHub-access-token>"
```

Use a fresh token. Never commit credentials, environment files, or access tokens.
**A previous token was committed to this repository. Revoke it in DagsHub immediately; editing this file does not erase Git history.**

## Streamlit deployment

Streamlit Community Cloud should use `streamlitapp.py` on the `main` branch.
It installs the lightweight, pinned packages in `requirements.txt`, including NumPy 2
and the scikit-learn 1.6.1 version used to save the current model artifacts.

For the separate MLOps training and Kubeflow workflows, install
`requirements-mlops.txt` instead.
