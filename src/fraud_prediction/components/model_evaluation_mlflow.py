import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from fraud_prediction.utils.common import save_json 
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from fraud_prediction.entity.config_entity import EvaluationConfig
from datetime import datetime

class Evaluation:
    def __init__(self, config: EvaluationConfig): 
        self.config = config
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(str(path))
    
    def _prepare_validation_data(self):
        """โหลดและเตรียมข้อมูลสำหรับวัดผล (ลอจิกเดียวกับตอนเทรน)"""
        data_dir = self.config.training_data
        print(f"DEBUG: กำลังค้นหาไฟล์ใน Path -> {os.path.abspath(data_dir)}")

        if str(data_dir).endswith('csv') and os.path.isfile(data_dir):
            target_file = data_dir
        
        else:
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

            if not csv_files:
                csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

            if not csv_files:
                raise FileNotFoundError(f"Not found file .csv in: {os.path.abspath(data_dir)}")
        
            target_file = csv_files[0]
        
        print(f" Found data file: {target_file}")
        df = pd.read_csv(target_file)
        
        # 1. Feature Engineering (ต้องเหมือนตอนเทรน)
        df['diff_new_old_balance'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['diff_new_old_destiny'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud','step_weeks', 'step_days'] 
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

        # Important to put dtype = int bc about error while doing One-Hot
        df = pd.get_dummies(df, columns=['type'], dtype=int)
        df = df.dropna()

        # 2. แยก X, y
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']

        # 3. Scaling (ในระบบจริงควรโหลด scaler ที่ Save ไว้มาใช้ แต่ตอนนี้ทำใหม่เพื่อ Test)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.X_valid = np.asarray(X_scaled).astype('float32')
        self.y_valid = np.asarray(y).astype('float32')

    def evaluation(self):
        # 1. Load model 
        self.model = self.load_model(self.config.path_of_model)
        self._prepare_validation_data()
        
        #Predic class (0 or 1)
        y_pred_prob = self.model.predict(self.X_valid)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Measure results by using Data Array
        self.recall = recall_score(self.y_valid, y_pred)
        self.precision = precision_score(self.y_valid, y_pred)
        self.f1 = f1_score(self.y_valid, y_pred)
        self.score = self.model.evaluate(self.X_valid, self.y_valid) # [loss, accuracy]
        
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self, experiment_name):
        """ส่งผลลัพธ์ขึ้น DagsHub/MLflow """
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.set_experiment(experiment_name)

        if mlflow.active_run():
            mlflow.end_run()

        # เช็คว่าเชื่อมต่อถูกที่ไหม
        print(f"Sending data to: {mlflow.get_tracking_uri()}")

        now = datetime.now().strftime("%Y%m%d_%H%M")
        auto_run_name = f"Initial_Run_After_Reset_{now}"

        with mlflow.start_run(run_name=auto_run_name, nested=True):
            params = self.config.all_params
            mlflow.log_params({
                "epochs": int(params.EPOCHS),
                "batch_size": int(params.BATCH_SIZE),
                "learning_rate": float(params.LEARNING_RATE),
                "num_features": int(params.NUM_FEATURES)
            })
            
            # บันทึก Metrics
            mlflow.log_metrics({
                "loss": float(self.score[0]), 
                "accuracy": float(self.score[1]),
                "recall": float(self.recall),     # บอกว่า "จับคนโกงได้กี่ % จากคนโกงทั้งหมด"
                "precision": float(self.precision), # บอกว่า "ที่ทายว่าโกง ใช่คนโกงจริงกี่ %"
                "f1_score": float(self.f1)
            })
            
            # 2. ทำ Signature เพื่อบอกโครงสร้างข้อมูลที่โมเดลต้องการ
            # ช่วยให้เวลาเอาไปใช้ต่อใน Stage 05 ไม่ต้องเดาว่าต้องใส่ Column อะไรบ้าง
            signature = infer_signature(self.X_valid, self.model.predict(self.X_valid))

            # บันทึกโมเดลขึ้น Cloud (Model Registry)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    model=self.model, 
                    artifact_path="model", 
                    registered_model_name="FraudDetection_Model", # ชื่อในหน้า Models
                    signature=signature
                )
                print("Saved data to DagsHub: Done")
            else:
                mlflow.keras.log_model(
                    #self.model, name="model"
                    model=self.model, 
                    artifact_path="model", 
                    signature=signature
                )