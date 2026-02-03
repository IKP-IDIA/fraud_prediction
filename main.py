from fraud_prediction import logger 
from fraud_prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fraud_prediction.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from fraud_prediction.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from fraud_prediction.pipeline.stage_04_model_evaluation import EvaluationPipeline
import os

os.environ["MLFLOW_TRACKING_USERNAME"]="ArtitayaN"
os.environ["MLFLOW_TRACKING_PASSWORD"]="d10f05c3bf59a9f946e535cddb121b4d48d5e8b9"

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
    model_trainer .main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e