import tensorflow as tf
import os
import mlflow
import dagshub
from fraud_prediction import logger 
from fraud_prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fraud_prediction.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from fraud_prediction.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from fraud_prediction.pipeline.stage_04_model_evaluation import EvaluationPipeline
from datetime import datetime


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ArtitayaN/fraud_prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="ArtitayaN"
os.environ["MLFLOW_TRACKING_PASSWORD"]="d10f05c3bf59a9f946e535cddb121b4d48d5e8b9"

EXPERIMENT_NAME = "Fraud_Detection_v1"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"Main_Pipeline_{datetime.now().strftime('%Y%m%d')}"):
  STAGE_NAME = "Data Ingestion stage"
  
  try: 
      logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
      obj = DataIngestionTrainingPipeline()
      obj.main() 
      logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  except Exception as e: 
      logger.exception(e) 
      raise e 
  
  STAGE_NAME = "Prepare base model"
  try:
      logger.info(f"***************") 
      logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
      prepare_base_model = PrepareBaseModelTrainingPipeline() 
      prepare_base_model.main() 
      logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  except Exception as e:
      logger.exception(e) 
      raise e
  
  STAGE_NAME = "Model Training"
  try:
      logger.info(f"*******************")
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
      model_trainer = ModelTrainingPipeline()
      model_trainer.main(experiment_name=EXPERIMENT_NAME)
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  except Exception as e:
      logger.exception(e)
      raise e
  
  STAGE_NAME = "Evaluation stage"
  try:
     logger.info(f"*******************")
     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
     model_evalution = EvaluationPipeline()
     model_evalution.main(experiment_name=EXPERIMENT_NAME)
     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  
  except Exception as e:
          logger.exception(e)
          raise e