import torch
import pandas as pd
from pathlib import Path

def save_model(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer = None,
               model_name: str = "model.pth",
               target_dir: str = "output/models/",
               epoch: int = None,
               val_acc: float = None):
  """
  Save a PyTorch model (and optionally optimizer, epoch, and val_acc) to a target directory.
  """
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  save_dict = {"model_state_dict": model.state_dict()}
  if optimizer is not None:
    save_dict["optimizer_state_dict"] = optimizer.state_dict()
  if epoch is not None:
    save_dict["epoch"] = epoch
  if val_acc is not None:
    save_dict["val_acc"] = val_acc

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(save_dict, model_save_path)

def save_results_to_csv(results: dict, filename: str, target_dir: str = "output/logs/"):
  """
  Saves the training/testing results dictionary to a CSV file in a specific directory.
  """
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)
  file_path = target_dir_path / filename
  df = pd.DataFrame(results)
  df.to_csv(file_path, index=False)
  print(f"[INFO] Results saved to: {file_path}")

def load_model(model_path: str, 
               model: torch.nn.Module, 
               optimizer: torch.optim.Optimizer = None,
               device: str = "cpu"):
  """
  Load a saved PyTorch model (and optionally optimizer, epoch, and val_acc).
  """
  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint["model_state_dict"])
  epoch = checkpoint.get("epoch", None)
  val_acc = checkpoint.get("val_acc", None)
  if optimizer is not None and "optimizer_state_dict" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  print(f"[INFO] Model loaded from: {model_path}")
  return model, optimizer, epoch, val_acc
