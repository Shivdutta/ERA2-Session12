import os
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate with your Kaggle credentials
api = KaggleApi()
api.authenticate()

# Specify the dataset you want to download
dataset_name = 'thehir0/mushroom-species'

# Specify the directory where you want to save the downloaded files
download_dir = '/home/shiv-nlp-mldl-cv/Downloads/Music/mushroom_dataset'

# Create the directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

print("Dataset started.")

# Download the dataset files
api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

print("Dataset downloaded and unzipped successfully.")
