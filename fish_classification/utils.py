import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir:str,
               model_name:str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".ph"), "Model name should end with .ph or .pth"

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model at {model_save_path}")

    torch.save(obj=model.state_dict(),f=model_save_path)