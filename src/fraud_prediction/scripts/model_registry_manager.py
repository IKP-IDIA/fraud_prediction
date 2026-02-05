import mlflow

class ModelRegistryManager:
    def __init__(self, model_name="FraudDetection_Model"):
        self.client = mlflow.tracking.MlflowClient()
        self.model_name = model_name

    def promote_to_production(self, version: int):
        """???????????? Stage ????????? Production"""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True # ?????????????????????
        )
        print(f"Model {self.model_name} version {version} is now in Production.")

# How to use:
# manager = ModelRegistryManager()
# manager.promote_to_production(version=1)