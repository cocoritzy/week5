import os
import shutil

current_folder = os.getcwd()

source_path = '/root/.cache/kagglehub/datasets/chrisfilo/urbansound8k/versions/1'
target_path = os.path.join(current_folder, 'UrbanSound8K')

# 📂 Recursively copy the dataset folder into your working directory
shutil.copytree(source_path, target_path)

print(f"Copied dataset to: {target_path}")
