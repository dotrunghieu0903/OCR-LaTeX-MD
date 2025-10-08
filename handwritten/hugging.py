from huggingface_hub import HfApi, Repository, login
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer
)

login(token='*')  # replace '*' with your actual token or use environment variables for security
api = HfApi()
model_name = "DGurgurov/im2latex"
api.create_repo(repo_id=model_name, exist_ok=True)

# cloning the repository
checkpoint_dir = f"checkpoints/checkpoint_epoch_6_step_19400"

# saving model and tokenizer
api.upload_folder(
    folder_path=checkpoint_dir,
    path_in_repo='',
    repo_id=model_name
)