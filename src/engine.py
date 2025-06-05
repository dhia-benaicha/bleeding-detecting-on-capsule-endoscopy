"""
Contains functions for training and testing a PyTorch model (binary classification, class imbalance aware).
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from utils import save_load  # Assumes save_checkpoint function exists

def train_step(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader, 
  loss_fn: torch.nn.Module, 
  optimizer: torch.optim.Optimizer,
  device: torch.device
) -> Tuple[float, float]:
  
  model.train()
  train_loss, train_acc = 0, 0

  for X, y in dataloader:
    X, y = X.to(device), y.to(device).float().view(-1, 1)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred) 

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc

def test_step(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader, 
  loss_fn: torch.nn.Module,
  device: torch.device
) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch (binary classification)."""
  model.eval()
  test_loss, test_acc = 0, 0
  
  with torch.inference_mode():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device).float().view(-1, 1)
      y_pred = model(X)
      print(f"y_pred: {y_pred.shape}, y: {y.shape}")
      loss = loss_fn(y_pred.unsqueeze(dim=1), y)
      test_loss += loss.item()
      test_pred_labels = y_pred.argmax(dim=1)
      test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  test_loss /= len(dataloader)
  test_acc /= len(dataloader)
  return test_loss, test_acc

def train(
  model: torch.nn.Module, 
  train_dataloader: torch.utils.data.DataLoader, 
  test_dataloader: torch.utils.data.DataLoader, 
  optimizer: torch.optim.Optimizer,
  loss_fn: torch.nn.Module,
  epochs: int,
  device: torch.device,
  logs_dir: str = "output/logs/",
  save_dir: str = "output/models/",
  model_name: str = "model.pth",
  save_best: bool = True
) -> Dict[str, List]:
  """Trains and tests a PyTorch model for a specified number of epochs, saves checkpoints."""
  results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
  model.to(device)
  best_acc = 0.0

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
    test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

    print(
      f"Epoch: {epoch+1} | "
      f"train_loss: {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
      f"test_loss: {test_loss:.4f} | "
      f"test_acc: {test_acc:.4f}"
    )

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    # save best model
    if save_best and test_acc > best_acc:
      best_acc = test_acc
      save_load.save_model(
        model=model,
        optimizer=optimizer,
        model_name=f"{model_name}.pth",
        target_dir=save_dir
      )

  save_load.save_results_to_csv(
  results=results,
  filename=f"{model_name}_results.csv",
  target_dir=logs_dir
  )

  return results 
