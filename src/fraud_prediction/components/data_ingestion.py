import os
import zipfile
import gdown
from fraud_prediction import logger
from fraud_prediction.utils.common import get_size
from fraud_prediction.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

     
    def download_file(self) -> str:
        try: 
            #dataset_url = self.config.onedrive_file_path
            zip_download_dir = self.config.local_data_file
            
            # Create destination directory 
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            file_id = "1xHzaCHq6gEXFmYN2Cc2Yb1jVxaAayb2y"

            if os.path.exists(self.config.local_data_file):
                os.remove(self.config.local_data_file)
                logger.info(f"Removed corrupted file : {zip_download_dir}")

            logger.info(f"Downloading data from : {file_id}")

            gdown.download(id=file_id, output=str(zip_download_dir), quiet=False)

            logger.info(f"Downloaded successfully into {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


