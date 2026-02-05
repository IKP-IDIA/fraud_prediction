import mlflow


with mlflow.start_run():
    mlflow.log_param("model_type", "test")
    mlflow.log_metric("accuracy", 0.95)
    print("Test run completed successfully!")