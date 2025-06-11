from torch.utils.tensorboard import SummaryWriter
import os

def log_results_tensorboard(results: dict, model_name: str, log_dir: str = "runs", start_epoch: int = 0):
    """
    Logs training and testing metrics to TensorBoard for a single model.

    Args:
        results (dict): Dictionary containing metric lists. 
            Expected keys: 'train_loss', 'test_loss', 'train_acc', 'test_acc'.
            Each value should be a list of metric values per epoch.
        model_name (str): Name of the model. Used as a subdirectory in log_dir.
        log_dir (str, optional): Parent directory for TensorBoard logs. Defaults to "runs".
        start_epoch (int, optional): Starting epoch number for logging. Defaults to 0.

    Example:
        results = {
            "train_loss": [0.5, 0.4],
            "test_loss": [0.6, 0.5],
            "train_acc": [0.8, 0.85],
            "test_acc": [0.75, 0.8]
        }
        log_results_tensorboard(results, "my_model")

    If you previously trained for 10 epochs, and now train for 5 more, call:
        log_results_tensorboard(results, "my_model", start_epoch=10)
    """
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    num_epochs = len(results["train_loss"])
    for i, epoch in enumerate(range(start_epoch, start_epoch + num_epochs)):
        writer.add_scalar("Loss/Train", results["train_loss"][i], epoch)
        writer.add_scalar("Loss/Test", results["test_loss"][i], epoch)
        writer.add_scalar("Accuracy/Train", results["train_acc"][i], epoch)
        writer.add_scalar("Accuracy/Test", results["test_acc"][i], epoch)
    writer.close()
    print(f"[INFO] TensorBoard logs updated for {model_name} in {os.path.join(log_dir, model_name)}")

def log_multiple_models_tensorboard(results_list: list, model_names: list, log_dir: str = "runs"):
    """
    Logs metrics for multiple models to TensorBoard.

    Args:
        results_list (list): List of results dictionaries, one per model.
            Each dictionary should have the same structure as described in log_results_tensorboard.
        model_names (list): List of model names corresponding to each results dictionary.
        log_dir (str, optional): Parent directory for TensorBoard logs. Defaults to "runs".

    Example:
        results_list = [results_model1, results_model2]
        model_names = ["model1", "model2"]
        log_multiple_models_tensorboard(results_list, model_names)
    """
    for results, name in zip(results_list, model_names):
        log_results_tensorboard(results, name, log_dir)