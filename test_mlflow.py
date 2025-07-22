# test_mlflow.py
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/rhitwik.2005/Colorectal_Cancer_Survival_Prediction.mlflow")
client = mlflow.tracking.MlflowClient()

experiments = client.list_experiments()
print("✅ Connected. Available experiments:")
for exp in experiments:
    print(f"- {exp.name} (ID: {exp.experiment_id})")
