import os
import pandas as pd
import numpy as np
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from fraud_prediction.entity.config_entity import TrainingConfig
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Training: 
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        """โหลด model ANN ที่เตรียมไว้จาก stage_02"""
        self.model = tf.keras.models.load_model(
            self.config.update_base_model_path
        )

        # Re-compile immediatly with new optimizer 
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model DownModel Downloaded and Re-Compiled already.")

    def prepare_data(self):
        """เตรียมข้อมูลตามลอจิก Notebook"""
        # 1. โหลดไฟล์ CSV
        data_dir = self.config.training_data
        csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"ไม่พบไฟล์ CSV ใน {data_dir}")

        df = pd.read_csv(csv_files[0])

        fraud_df = df[df['isFraud'] == 1]
        normal_df = df[df['isFraud'] == 0]
        
        ratio = self.config.params_sampling_ratio
        n_normal=len(fraud_df)*ratio 
        # ตรวจสอบว่ามีข้อมูลปกติพอให้สุ่มไหม
        n_normal = min(n_normal, len(normal_df)) 
        
        normal_downsampled = normal_df.sample(n=n_normal, random_state=42)
        
        df = pd.concat([fraud_df, normal_downsampled])
        df = df.sample(frac=1, random_state=42)

        print(f"ข้อมูลที่ใช้เทรน (Ratio 1:{ratio}): ทั้งหมด {df.shape[0]} แถว")

        # 2. Feature Engineering
        df['diff_new_old_balance'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['diff_new_old_destiny'] = df['newbalanceDest'] - df['oldbalanceDest']

        # 3. Feature Selection & One-Hot Encoding
        cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud','step_weeks', 'step_days'] 
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

        # ป้องกัน NotImplementedError ด้วย dtype=int
        df = pd.get_dummies(df, columns=['type'], dtype=int)
        df = df.dropna()

        # 4. แยก Feature และ Target (isFraud)
        target_col = 'isFraud'
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 5. Seperate Data and Scaling
        X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_valid_scaled = scaler.transform(X_valid_raw)

        # 6. บันทึกผลลัพธ์ลงใน Class Attribute (แปลงเป็น float32 ทันที)
        self.X_train = np.asarray(X_train_scaled).astype('float32')
        self.X_valid = np.asarray(X_valid_scaled).astype('float32')
        self.y_train = np.asarray(y_train_raw).astype('float32')
        self.y_valid = np.asarray(y_valid_raw).astype('float32')
        
        print(f" Prepared Data Done and จำนวน Features สุดท้าย: {self.X_train.shape[1]}")

    def train(self):
        """เริ่มเทรนโมเดล ANN (เวอร์ชันสมบูรณ์สำหรับข้อมูลตาราง)"""
        # 1. แปลงเป็น Tensor เพื่อประสิทธิภาพและความเสถียรบน TensorFlow
        X_train_tensor = tf.convert_to_tensor(self.X_train, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(self.y_train, dtype=tf.float32)
        X_valid_tensor = tf.convert_to_tensor(self.X_valid, dtype=tf.float32)
        y_valid_tensor = tf.convert_to_tensor(self.y_valid, dtype=tf.float32)

        print(f"เริ่มต้นการเทรนด้วยข้อมูล Shape: {X_train_tensor.shape}")

        # 2. Start to train 
        self.history = self.model.fit(
            X_train_tensor,
            y_train_tensor,
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size,
            validation_data=(X_valid_tensor, y_valid_tensor),
            verbose=1
        )

        # 3. Save model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))