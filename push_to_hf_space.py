from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO_ID = "harikrishna1985/predictive-maintenance-space"

FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "config/config.yaml",
    "src/predict.py",
]

api = HfApi(token=HF_TOKEN)

for file_path in FILES_TO_UPLOAD:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
    )

print("All deployment files uploaded to Hugging Face Space.")
