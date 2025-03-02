import os
import requests
import zipfile
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "data"

if image_path.is_dir():
    print(f"{image_path} already exists")
else:
    with zipfile.ZipFile(data_path / "data.zip", "r") as zip_ref:
        print("Unzipping the file ...")
        zip_ref.extractall(image_path)

print(f"data is at {image_path}, deleting the zip file")
os.remove(data_path / "data.zip")